"""Expected-value decision engine on top of the buyer belief (plan v3, S4).

Each turn it re-plans (receding horizon): candidate concession paths from now to the final round
are scored against particles drawn from the belief, and the first step of the best path is sent.
A path's value per particle follows the negotiation forward:
  - the buyer's counter at round j reflects willingness(j-1); we accept it if it covers our ask;
  - our offer x at round j is accepted if x <= willingness(j);
  - a rejected offer ends in a walk (score 0) once j >= T for a buyer that walks, or in a
    deadlock (-0.25) after the final round.
The trade-off findings enter as maths, not rules (tradeoff-analysis.md section 7):
deadlocks cost 0.25 (G2), the chance of leaving pulls prices down early (G3), a slow concession
shape keeps them up (G4), overshooting is priced through the acceptance probability (G5), and
walking is considered only at the final turn (G1). The fallback's own path is always a candidate.

Tags are not used here yet (variant V1); S5 feeds classifier tags into the belief.
"""
import math

from negotiator.belief import BuyerBelief, seed_for
from negotiator.fallback import TEXT_ACCEPT, TEXT_OFFER, TEXT_WALK, FallbackEngine, _is_price, norm_sku, norm_qty

DEADLOCK = 0.25


class BeliefEngine(FallbackEngine):
    def __init__(self, store, policy, n_plan=120, grid=10, model_sd=0.05, use_tags=False,
                 fallback_floor=False, leave_exempt=0.2, floor_rounds=None, final_cap=True,
                 bid_gap=False, accept_guard=False, bid_anchor=False, limit_prior="uniform", text_type=False,
                 bid_lower=False, sd_mode="const", sigmas=None, closed_final=False, probe=False,
                 e_n=12, cap_q=None, final_q=False, hold=False, t_prior=None, no_rise=False, **fallback_params):
        """fallback_floor (refine, 2026-11-21): never ask below the fallback's price for the same round,
        unless the belief gives at least `leave_exempt` chance that the buyer walks after this round.
        S6 found the unconstrained planner under-prices LLM buyers, who bid low but accept far above their
        bids (margin lost about 6 points vs the fallback's 2; results/eval-report.md).
        floor_rounds: apply that floor only up to this round (None = every round). A floor on every round
        failed D8 on the noisy suites: high asks make rejections uninformative and the final-turn belief
        too pessimistic (results/refine-report.md)."""
        super().__init__(store, policy, **fallback_params)
        self.n_plan, self.grid, self.model_sd, self.use_tags = n_plan, grid, model_sd, use_tags
        self.fallback_floor, self.leave_exempt, self.floor_rounds = fallback_floor, leave_exempt, floor_rounds
        self.final_cap = final_cap        # G5 cap on the last offer; False only for diagnosis (bench/diagnose.py)
        # Fix 1 (2026-11-22, results/bottleneck-report.md: accepting bids below what the buyer would pay):
        #   bid_gap      plan with the buyer accepting up to bid + gap, the gap learned from rejections
        #   accept_guard accept a bid only if it reaches our own ask this round (or final turn / likely to leave)
        #   bid_anchor   also consider counters a little above the buyer's bid
        self.bid_gap, self.accept_guard, self.bid_anchor = bid_gap, accept_guard, bid_anchor
        self.limit_prior = limit_prior    # identification fix: "floor" anchors the limit prior to our floor
        self.text_type = text_type        # identification fix: manipulation text is evidence of a stubborn buyer
        self.bid_lower = bid_lower        # identification fix: the limit is at least the buyer's highest bid
        self.sd_mode = sd_mode            # identification fix: "remaining" shrinks the model-error spread late
        self.sigmas = tuple(sigmas) if sigmas else None   # bid-noise levels of the belief grid (None = default)
        # Information-seeking (2026-11-23, results/identification-report.md section 7):
        #   closed_final  score a plan with the final turn decided on what the plan's earlier offers would reveal:
        #                 among the sampled buyers that reach the final turn undealt, the seller walks or picks the
        #                 final price that is best on average for that group (it cannot tell them apart). The
        #                 open-loop score makes the final offer to every sampled buyer, so it never values a plan
        #                 whose earlier offers sort feasible buyers (deal early) from infeasible ones (walk at the end).
        #   probe         add candidate plans that test the floor one round before the end
        self.closed_final, self.probe = closed_final, probe
        # Endgame that follows identification (bottleneck-report.md counterfactuals F and G):
        #   e_n      shape-grid size of the belief (12 = original)
        #   cap_q    final-offer cap = max(floor, best bid, the cap_q quantile of the buyer's effective limit), so a
        #            confident belief may ask above the buyer's last bid at the end (None = the G5 cap)
        #   final_q  add final-price candidates at quantiles of what the sampled buyers would accept at the end
        self.e_n, self.cap_q, self.final_q = e_n, cap_q, final_q
        #   hold     add "hold, then drop" plans: ask the current cap for h rounds, then a lower price to the end.
        #            Late concessors reach their limit only in the last rounds, so holding high and dropping late
        #            beats a straight line (counterfactual G)
        self.hold = hold
        self.t_prior = t_prior        # deadline prior (belief.t_weights); None = uniform over the grid
        # no_rise (bug fix, 2026-11-23): rounding a plan's price up to whole rupees can put the offer a fraction of a
        # rupee above our previous ask; the old guard then sent the fallback schedule's offer, which can be far
        # higher (an ask that rises mid-negotiation, in about 10% of kit negotiations). With no_rise the previous
        # offer is repeated instead, so asks really never rise.
        self.no_rise = no_rise

    def start(self, negotiation_id):
        super().start(negotiation_id)
        self.belief = None
        self.offers = {}                  # round -> our offered share of list
        self.ask = None                   # our latest offered share (asks never rise)
        self.prev_items = None            # our latest offer's lines (no_rise)
        self.last_plan = None
        self._vcache = None               # (list total, cost total) of the basket, once it is known

    # ------------------------------------------------------------ basket figures
    def _totals(self):
        L = sum(q * self.store[s]["price"] for s, q in self.basket.items())
        C = sum(q * self.store[s]["cost"] for s, q in self.basket.items())
        F = sum(q * self.floors[s] for s, q in self.basket.items())
        return L, C, F

    def value(self, share):
        """Score value of a deal at `share` of the basket's list total (profit / list-price profit)."""
        if self._vcache is None:
            L, C, _ = self._totals()
            self._vcache = (L, C)
        L, C = self._vcache
        return (share * L - C) / (L - C) if L > C else 0.0

    def bid_share(self, items):
        """The buyer's bid as a share of list over our basket lines it prices validly, or None."""
        num = den = 0
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            s, q, v = norm_sku(it.get("sku"), self.store), norm_qty(it.get("qty")), it.get("unit_price")
            if s is None or s not in self.basket or q is None or not _is_price(v):
                continue
            q = min(q, self.basket[s])
            num, den = num + q * v, den + q * self.store[s]["price"]
        return num / den if den else None

    def allocate(self, share):
        """Per-line integer prices for a target share of the list total.

        Every line moves the same fraction of the way from its own floor to its list price, so the
        floor share prices every line exactly at its floor and share 1 prices every line at list.
        (Spreading proportionally to list instead pushes low-floor lines above their floor and makes
        the "floor" offer overshoot buyers whose limit sits just above it.)
        """
        L, _, F = self._totals()
        lam = 0.0 if L <= F else min(1.0, max(0.0, (share * L - F) / (L - F)))
        return [{"sku": s, "qty": q,
                 "unit_price": min(self.store[s]["price"],
                                   math.ceil(self.floors[s] + lam * (self.store[s]["price"] - self.floors[s])))}
                for s, q in self.basket.items()]

    def share_of(self, items):
        L, _, _ = self._totals()
        return sum(i["qty"] * i["unit_price"] for i in items) / L

    # ------------------------------------------------------------ turn
    def _decide(self, msg, t, r):
        L, C, F = self._totals()
        floor_share = F / L
        bid = self.bid_share(msg.get("items")) if t in ("request", "counter") else None
        opened_now = False
        if self.belief is None:
            opening = bid if t == "request" else None
            self.belief = BuyerBelief(floor_share, self.max_rounds, opening, self.model_sd, self.limit_prior,
                                      self.text_type, self.bid_lower, self.sd_mode,
                                      e_n=self.e_n, t_prior=self.t_prior, **({"sigmas": self.sigmas} if self.sigmas else {}))
            self.seed = seed_for(sorted(self.basket.items()), round(opening, 4) if opening is not None else None)
            opened_now = opening is not None
        if r - 1 in self.offers and t == "counter":
            self.belief.observe_rejection(r - 1, self.offers[r - 1])
        if bid is not None and not opened_now:
            self.belief.observe_bid(r - 1, bid)
        self._observe_text(msg, r, bid)
        self.belief.update()

        fb_items = super()._decide(msg, t, r)                       # the fallback's view of this turn
        cap = 1.0 if self.ask is None else self.ask
        fb_path = [min(cap, self.share_of(self.allocate_at(j))) for j in range(r, self.max_rounds + 1)]
        # G5 (tradeoff-analysis T4): the last offer goes above the floor only as far as the buyer's own
        # bids already reach; an overshoot there loses the deal and can deadlock.
        self.last_cap = max(floor_share, max((x for _, x in self.belief.bids), default=floor_share)) \
            if self.final_cap else 1.0
        if self.final_cap and self.cap_q is not None:
            self.last_cap = max(self.last_cap, self.belief.limit_estimate(self.cap_q, self.cap_q, effective=True)[1])
        starts =[min(cap, max(floor_share, bid + d)) for d in (0.02, 0.05, 0.10)] \
            if self.bid_anchor and bid is not None else []
        plans = [p[:-1] + [min(p[-1], self.last_cap)]
                 for p in self.candidate_plans(r, floor_share, cap, fb_path, starts)]
        self.leave_risk = self.belief.p_leave_by(r)
        if self.fallback_floor and self.leave_risk < self.leave_exempt:
            # concede no faster than the fallback's schedule unless the buyer looks likely to leave now
            last = self.max_rounds if self.floor_rounds is None else self.floor_rounds
            plans = [[max(x, f) if r + k <= last else x for k, (x, f) in enumerate(zip(p, fb_path))] for p in plans]
        sample = self.planning_sample(r)
        self._cap_now = cap
        if self.final_q and r >= self.max_rounds:
            acc = sorted(min(cap, self.last_cap, wacc[r]) for _, _, _, wacc in sample)
            extra = {acc[int(q * (len(acc) - 1))] for q in (0.1, 0.2, 0.3, 0.4, 0.5)}
            plans += [[x] for x in extra if x >= floor_share]
        best_ev, best = -1e9, None
        for plan in plans:
            ev = self.plan_value(plan, r, sample, floor_share)
            if ev > best_ev:
                best_ev, best = ev, plan
        self.last_plan = (best, best_ev)

        valid = self.valid_bid(msg.get("items")) if t in ("request", "counter") else None
        if valid:
            valid_share = sum(self.basket[s] * valid[s] for s in self.basket) / L
            bid_value = self.value(valid_share)
            guarded = self.accept_guard and r < self.max_rounds and self.leave_risk < self.leave_exempt \
                and valid_share < best[0] - 1e-9          # below our own ask this round: counter instead
            if bid_value >= best_ev and not guarded:
                return {"type": "accept", "text": TEXT_ACCEPT}
        if r >= self.max_rounds and best_ev < 0:
            return {"type": "walk_away", "text": TEXT_WALK}
        items = self.allocate(best[0])
        if self.share_of(items) > cap + 1e-9:
            if self.no_rise and self.prev_items:
                items = [dict(i) for i in self.prev_items]
            elif fb_items.get("type") == "offer":
                items = fb_items["items"]
        self.prev_items = [dict(i) for i in items]
        self.ask = self.share_of(items)
        self.offers[r] = self.ask
        return {"type": "offer", "items": items, "text": TEXT_OFFER}

    def _observe_text(self, msg, r, bid):
        """Hook for text evidence (tags) before the belief update. V1 reads no text."""

    def allocate_at(self, j):
        p = self.prices(j)
        return [{"sku": s, "qty": q, "unit_price": p[s]} for s, q in self.basket.items()]

    # ------------------------------------------------------------ planning
    def candidate_plans(self, r, floor_share, cap, fb_path, extra_starts=()):
        n = self.max_rounds - r + 1
        plans = [fb_path]
        gap = max(0.0, cap - floor_share)
        starts = [floor_share + gap * (g / (self.grid - 1) if self.grid > 1 else 1.0) for g in range(self.grid)]
        for x0 in starts + list(extra_starts):
            if self.probe and n >= 3:
                plans.append([x0] * (n - 2) + [floor_share, floor_share])
                plans.append([x0 + (floor_share - x0) * k / (n - 2) for k in range(n - 1)] + [floor_share])
            for frac in (0.0, 0.25, 0.5):
                end = min(x0, floor_share + gap * frac)
                if n == 1:
                    plans.append([x0])
                    break
                plans.append([x0 + (end - x0) * k / (n - 1) for k in range(n)])
        if self.hold:
            for h in range(1, n):
                for g in range(self.grid):
                    plans.append([cap] * h + [floor_share + gap * g / max(1, self.grid - 1)] * (n - h))
        return plans

    def plan_value(self, plan, r, sample, floor_share):
        if self.closed_final and len(plan) > 1:
            return self.plan_value_closed(plan, r, sample, floor_share)
        M = self.max_rounds
        total = 0.0
        for T, stub, w, wacc in sample:          # w: the buyer's bid curve; wacc: what it accepts (w + gap)
            outcome = 0.0
            for k, x in enumerate(plan):
                j = r + k
                if k > 0 and w[j - 1] >= x and w[j - 1] >= floor_share:
                    outcome = self.value(w[j - 1])
                    break
                if x <= wacc[j]:
                    outcome = self.value(x)
                    break
                if j >= T and not stub:
                    break
                if j >= M:
                    outcome = -DEADLOCK
                    break
            total += outcome
        return total / len(sample)

    def plan_value_closed(self, plan, r, sample, floor_share):
        """plan_value with the final turn chosen for the group of sampled buyers that reach it undealt."""
        M = self.max_rounds
        total, reach = 0.0, []
        for T, stub, w, wacc in sample:
            ended = False
            for k, x in enumerate(plan[:-1]):
                j = r + k
                if k > 0 and w[j - 1] >= x and w[j - 1] >= floor_share:
                    total += self.value(w[j - 1])
                    ended = True
                    break
                if x <= wacc[j]:
                    total += self.value(x)
                    ended = True
                    break
                if j >= T and not stub:
                    ended = True                                   # the buyer walks: 0
                    break
            if not ended:
                reach.append((T, stub, w, wacc))
        if reach:
            # the final round's counter reflects w(M - 1); a valid bid is accepted whatever the final choice
            finals = {plan[-1], floor_share}
            if self.final_q:
                top = min(self._cap_now, self.last_cap)
                acc = sorted(min(top, wacc[M]) for _, _, _, wacc in reach)
                finals |= {acc[int(q * (len(acc) - 1))] for q in (0.1, 0.25, 0.5)}
                finals = {x for x in finals if x >= floor_share} or {floor_share}
            finals = sorted(finals)
            best = None
            for xf in [None] + finals:                                  # None = walk (still accept a valid bid)
                v = 0.0
                for T, stub, w, wacc in reach:
                    b = w[M - 1]
                    if b >= floor_share and (xf is None or b >= xf):
                        v += self.value(b)
                    elif xf is not None and xf <= wacc[M]:
                        v += self.value(xf)
                    elif xf is not None and (stub or M < T):
                        v -= DEADLOCK
                best = v if best is None else max(best, v)
            total += best
        return total / len(sample)

    def planning_sample(self, r):
        """Draws from the belief as (T, stub, bid curve, acceptance curve), computed once per turn.

        Without the bid-gap model the two curves are the same. With it, draw k gets gap gaps[k % len(gaps)]
        from the values the rejections still allow (deterministic, so identical negotiations plan alike)."""
        b = self.belief
        a = b.opening
        gaps = b.bid_gap_values(self.bid_gap if isinstance(self.bid_gap, str) else "uniform") if self.bid_gap else [0.0]
        out = []
        for k, (cell, R) in enumerate(b.sample(self.n_plan, self.seed + r)):
            g = cell.gs(b.j0)
            w = [a + (R - a) * g[j] for j in range(self.max_rounds + 1)]
            gap = gaps[k % len(gaps)]
            out.append((cell.T, cell.stub, w, [min(1.0, x + gap) for x in w] if gap else w))
        return out
