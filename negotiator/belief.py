"""Bayesian buyer model (plan v3, S4): a grid over how a buyer concedes, with its limit integrated exactly.

Concession model (prices as a share of the basket's list total):

    willingness(j) = a + (R - a) * g(j),   g(j) = min(1, j / T) ** e

a: opening level (the opening bid), R: the private limit, T: the round by which it reaches R (and,
if it is not stubborn, after which it walks), e: concession shape (e > 1 concedes late),
stub: never walks on its own, sigma: bid noise. The buyer accepts our offer x at round j when
x <= willingness(j). The family is broad on purpose; it describes "concedes towards a private
limit", not any one bot.

Because willingness is linear in R, for each cell (T, e, sigma, stub) of a fixed grid the
posterior over R is an exact truncated normal (uniform prior on [a, 1]), and each cell's marginal
likelihood is closed-form. Evidence:
  - bids: the counter received at round k reflects willingness(k - 1); the opening request is j = 0;
  - rejected offers: our offer x at round j was countered, so R < a + (x - a) / g(j) (upper bound);
  - survival: the buyer did not walk after round j, so not (j >= T and not stub);
  - text tags (plan v3, D4): claims reweight cells with a per-buyer reliability (a Beta counter).
    A "final offer" followed by a higher bid is a bluff and lowers it; manipulation tags lower it;
    claims that hold raise it. Tags never set prices.
A grid instead of random particles means no particle collapse, and identical evidence always
gives identical beliefs (plan v3, D7).
"""
import math
import random
from statistics import NormalDist

T_GRID = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15)            # >= 11 means "never reaches it in time"
def e_grid(n=12):
    """n concession shapes, log-spaced over [0.4, 6]."""
    return tuple(math.exp(math.log(0.4) + k * (math.log(6.0) - math.log(0.4)) / (n - 1)) for k in range(n))


E_GRID = e_grid(12)
SIGMAS = (0.01, 0.025, 0.06)
P_STUB = 0.5
SURVIVE_EPS = 0.03
REJECT_SLACK = 0.004
BID_SLACK = 0.005
SD_MIN_SHARE = 0.25
DEFAULT_OPENING = 0.6
# Prior on the deadline T (identification, 2026-11-23). None: uniform over T_GRID, so 70% of the mass says the buyer's
# willingness peaks before our last round. "lateX": X of the mass spread evenly over T <= 9, the rest over T >= 10
# (README.md:104-107: of the three public types only the cooperative one leaves before round 10).
def t_weights(prior):
    if not prior:
        return {T: 1.0 / len(T_GRID) for T in T_GRID}
    early_share = float(prior[4:]) / 100.0
    early = [T for T in T_GRID if T < 10]
    late = [T for T in T_GRID if T >= 10]
    return {T: (early_share / len(early) if T < 10 else (1.0 - early_share) / len(late)) for T in T_GRID}
CLAIM_TAGS = ("final_offer", "walk_threat", "budget_claim")
DISTRUST_TAGS = ("manipulation", "injection", "authority_claim", "false_agreement", "competitor_quote")
_N = NormalDist()
_SQRT_HALF = 0.7071067811865476


def _phi(x):
    """Standard normal CDF (math.erf directly; NormalDist.cdf is several times slower)."""
    return 0.5 * (1.0 + math.erf(x * _SQRT_HALF))


def seed_for(*parts):
    """A stable integer seed from the negotiation's content (Python's hash() is salted per run)."""
    import hashlib
    return int(hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16], 16)


_G_CACHE = {}


STUB_LR = 2.0
STUB_SIGNALS_MAX = 3
GAP_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
# Prior weight per gap value, as draws out of 20. A uniform prior lost 3-4 points on the kit bots and proxy suite,
# whose gap is 0: until our offers come within the gap of the bids, rejections say nothing, so the seller held
# high and cooperative buyers walked (results/fix1-report.md).
GAP_PRIOR = {"uniform": (4, 4, 4, 4, 4), "cautious": (14, 3, 2, 1, 0)}
GAP_SLACK = 0.01      # bids are rounded (kit bots to ₹10) and LLM bids are noisy
MODEL_SD = 0.05      # prediction-only spread for model error: real buyers do not follow this curve exactly;
                      # 0.05 picked by the D8 rule over 0.0 / 0.03 / 0.08 (results/S4-complete.md)


# Prior on the buyer's limit R (identification fix, 2026-11-22). "uniform": R uniform on [opening bid, list], the
# original. "floor": anchored to our floor share F, following the problem statement (most limits sit at or above
# what policy lets us sell for; some sit below, "and the right move is to walk away"; README.md:100-102):
# (R from, R to, probability) with bounds as multiples of F, and 1.0 meaning list price.
LIMIT_PRIORS = {
    "floor": ((None, 0.8, 0.05),          # well below the floor, kept possible (None = the buyer's opening bid)
              (0.8, 1.0, 0.20),           # just below the floor: infeasible buyers
              (1.0, None, 0.75)),         # between the floor and list (None = list)
    # less optimistic variants, for buyer mixes with more infeasible cases (the proxy suites are ~61% feasible)
    "floor65": ((None, 0.8, 0.08), (0.8, 1.0, 0.27), (1.0, None, 0.65)),
    "floor55": ((None, 0.8, 0.10), (0.8, 1.0, 0.35), (1.0, None, 0.55)),
    # "floor_low": the same 25% below the floor, but above it most limits sit close to the floor, as in the kit's
    # generator (buyers.py:103-158: feasible zones 0.05-0.6 / 0.1-0.7 / 0.4-1.0 of the floor-to-list gap) and the
    # proxies (0.005-0.3). A string bound "gX" means F + X * (1 - F).
    "floor_low": ((None, 0.8, 0.05), (0.8, 1.0, 0.20), (1.0, "g0.3", 0.40), ("g0.3", "g0.6", 0.22),
                  ("g0.6", None, 0.13)),
    # between "floor" (uniform above the floor) and "floor_low"
    "floor_mid": ((None, 0.8, 0.05), (0.8, 1.0, 0.20), (1.0, "g0.3", 0.32), ("g0.3", "g0.6", 0.24),
                  ("g0.6", None, 0.19)),
}


def _bound(b, F, default):
    if b is None:
        return default
    if isinstance(b, str):                        # "gX": X of the way from the floor to list
        return F + float(b[1:]) * (1.0 - F)
    return b * F


def prior_segments(a, F, prior):
    """Prior on theta = R - a as ascending (lo, hi, density) segments, from the opening bid a and floor share F."""
    if prior == "uniform" or prior not in LIMIT_PRIORS:
        width = max(1e-6, 1.0 - a)
        return [(0.0, width, 1.0 / width)]
    segs = []
    for lo_m, hi_m, w in LIMIT_PRIORS[prior]:
        lo_r = max(a, _bound(lo_m, F, a))
        hi_r = min(1.0, _bound(hi_m, F, 1.0))
        if hi_r > lo_r + 1e-9:
            segs.append([lo_r - a, hi_r - a, w])
    total = sum(s[2] for s in segs) or 1.0
    return [(lo, hi, w / total / (hi - lo)) for lo, hi, w in segs] or [(0.0, max(1e-6, 1.0 - a), 1.0)]


def effective_segments(eff, gM, width):
    """Map a prior on the effective limit's theta_eff = theta * gM (what the buyer pays by our last round) to a prior
    on theta, keeping theta <= width (R <= list). Not renormalised: a cell that cannot reach a limit the prior finds
    likely loses prior mass (conditioning on R <= list), which is what separates a late deadline from a high limit."""
    if gM >= 1.0 - 1e-9:
        return eff
    if gM <= 1e-9:
        return [(0.0, width, sum(d * (hi - lo) for lo, hi, d in eff if lo <= 1e-12) / width or 1e-9)]
    out = []
    for lo, hi, d in eff:
        lo_t, hi_t = lo / gM, min(width, hi / gM)
        if hi_t > lo_t + 1e-12:
            out.append((lo_t, hi_t, d * gM))
    return out or [(0.0, width, 1e-9)]


class Cell:
    __slots__ = ("T", "e", "sigma", "stub", "logw", "w", "mu", "sd", "lo", "hi", "z", "segs")

    def __init__(self, T, e, sigma, stub):
        self.T, self.e, self.sigma, self.stub = T, e, sigma, stub

    def gs(self, j0=0, upto=12):
        """g(j) for j = 0..upto, relative to the anchor round j0 (cached across negotiations: the grid is fixed)."""
        key = (self.T, self.e, j0, upto)
        if key not in _G_CACHE:
            G = [min(1.0, k / self.T) ** self.e for k in range(upto + 1)]
            g0 = G[j0]
            _G_CACHE[key] = [1.0 if g0 >= 1.0 else max(0.0, (G[k] - g0) / (1.0 - g0)) for k in range(upto + 1)]
        return _G_CACHE[key]

    def g(self, j, j0=0):
        return self.gs(j0)[min(j, 12)]

    # Posterior over theta = R - a: the likelihood N(mu, sd) times the piecewise-constant prior `segs`
    # (lo, hi, density), already truncated by rejections. Predictions widen sd by the model-error spread m
    # (prediction only; the likelihood uses the exact sd). With sd = 0 (no informative bid yet) the posterior
    # is the prior itself. With the "uniform" prior there is one segment and this equals the original maths.
    def _sdp(self, m):
        return math.sqrt(self.sd * self.sd + m * m) if self.sd > 0 else 0.0

    def _part(self, seg, x, sdp):
        """Unnormalised posterior mass of segment `seg` below x."""
        lo, hi, d = seg
        x = min(x, hi)
        if x <= lo:
            return 0.0
        if sdp <= 0:
            return d * (x - lo)
        return d * max(0.0, _phi((x - self.mu) / sdp) - _phi((lo - self.mu) / sdp))

    def cdf(self, theta, m=0.0):
        sdp = self._sdp(m)
        total = sum(self._part(s, s[1], sdp) for s in self.segs)
        if total <= 1e-300:                                  # posterior far outside every segment
            return 0.0 if theta < min(max(self.mu, self.segs[0][0]), self.segs[-1][1]) else 1.0
        return min(1.0, max(0.0, sum(self._part(s, theta, sdp) for s in self.segs) / total))

    def inv(self, u, m=0.0):
        sdp = self._sdp(m)
        masses = [self._part(s, s[1], sdp) for s in self.segs]
        total = sum(masses)
        if total <= 1e-300:
            return min(max(self.mu, self.segs[0][0]), self.segs[-1][1])
        target = u * total
        for (lo, hi, d), ms in zip(self.segs, masses):
            if target <= ms or (lo, hi, d) == self.segs[-1]:
                if sdp <= 0:
                    return min(hi, lo + target / d)
                p = _phi((lo - self.mu) / sdp) + target / d
                return min(hi, max(lo, self.mu + sdp * _N.inv_cdf(min(1 - 1e-12, max(1e-12, p)))))
            target -= ms
        return self.segs[-1][1]


class BuyerBelief:
    def __init__(self, floor_ratio, max_rounds=10, opening=None, model_sd=MODEL_SD, limit_prior="uniform",
                 text_type=False, bid_lower=False, sd_mode="const", sigmas=SIGMAS, e_n=12, t_prior=None):
        self.bid_lower, self.sd_mode = bid_lower, sd_mode
        self.log_t = {T: math.log(w) for T, w in t_weights(t_prior).items()}
        self.floor_ratio, self.max_rounds, self.model_sd = floor_ratio, max_rounds, model_sd
        self.limit_prior = limit_prior
        # Identification fix: manipulation / attack text is evidence of a buyer that never walks away on its own
        # ("manipulator never leaves on its own", README.md:107). Each tagged message multiplies the odds of the
        # stubborn cells by STUB_LR, for at most STUB_SIGNALS_MAX messages.
        self.text_type, self.stub_signals = text_type, 0
        self.a, self.j0 = opening, 0
        # e_n > 12 (identification fix): a finer shape grid, so noise-free bids pin the limit instead of being fit by
        # the nearest of 12 shapes (kit shapes 2.0 and 2.5 fall between grid points 1.75 / 2.24 / 2.87)
        es = E_GRID if e_n == 12 else e_grid(e_n)
        self.cells = [Cell(T, e, s, st) for T in T_GRID for e in es for s in sigmas for st in (False, True)]
        self.bids, self.rejections, self.survived = [], [], set()
        self.claims = []                                    # [round, tag, bid_share, value_share, status]
        self.alpha, self.beta = 1.0, 1.0
        self.update()

    @property
    def reliability(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def opening(self):
        return self.a if self.a is not None else DEFAULT_OPENING

    # ------------------------------------------------------------ evidence
    def observe_bid(self, j, share):
        """A priced counter reflecting willingness at round j (j = round received - 1)."""
        if self.a is None:                                  # no priced opening: the first bid anchors
            self.a, self.j0 = share, j
            return
        for c in self.claims:
            if c[4] == "open" and c[1] == "final_offer" and j > c[0]:
                if share > c[2] + 0.005:
                    c[4], self.beta = "bluff", self.beta + 1.0
                else:
                    c[4], self.alpha = "held", self.alpha + 1.0
        self.bids.append((j, share))

    def observe_rejection(self, j, offer_share):
        """Our offer at round j was countered rather than accepted, and the buyer stayed."""
        self.rejections.append((j, offer_share))
        self.survived.add(j)

    # ------------------------------------------------------------ bid gap (fix 1, 2026-11-22)
    # A buyer may accept an offer some way above its own counter: the "bid gap" g, so it accepts x at round j
    # when x <= bid(j) + g. Kit bots have g = 0; S6's LLM buyers accepted offers about 0.17 of list above their
    # last bid (results/bottleneck-report.md). Each round where our offer x_j was rejected and the buyer then
    # bid b_j (its willingness at j) proves g < x_j - b_j, so rejections close to the bid reveal a small gap.
    def bid_gap_bound(self):
        """The tightest proven upper bound on the gap, or None before any (offer, counter) pair."""
        bids = dict(self.bids)
        return min((x - bids[j] for j, x in self.rejections if j in bids), default=None)

    def bid_gap_values(self, prior="uniform"):
        """Gap values (share of list) still consistent with every rejection, each repeated by its prior weight,
        so cycling through the list draws from the prior restricted to the allowed values."""
        ub = self.bid_gap_bound()
        vals = [g for g, wgt in zip(GAP_GRID, GAP_PRIOR[prior]) for _ in range(wgt)
                if ub is None or g < ub + GAP_SLACK]
        return vals or [0.0]

    def observe_tags(self, j, tags, bid_share=None, value_share=None, threshold=0.5):
        """Classifier tag probabilities for the buyer's message at round j (stubbed in S4)."""
        if any(p >= threshold and tag in DISTRUST_TAGS for tag, p in (tags or {}).items()):
            self.stub_signals += 1                       # one signal per message, however many tags it carries
        for tag, p in (tags or {}).items():
            if p < threshold:
                continue
            if tag in DISTRUST_TAGS:
                self.beta += 0.5 * p
            elif tag in CLAIM_TAGS:
                self.claims.append([j, tag, bid_share if bid_share is not None else -1.0, value_share, "open"])

    # ------------------------------------------------------------ inference
    def update(self):
        a, rho = self.opening, self.reliability
        self._jl = min(12, max(j for j, _ in self.bids)) if self.bids else None   # last bid round, for _m
        width = max(1e-6, 1.0 - a)
        # "floor_eff": the "floor" table on the effective limit. The eff-space segments are capped at list (theta_eff
        # <= width); effective_segments then caps theta itself at width.
        eff_prior = self.limit_prior.endswith("_eff")          # e.g. "floor_eff": the "floor" table on w(M)
        base = prior_segments(a, self.floor_ratio, self.limit_prior[:-4] if eff_prior else self.limit_prior)
        prior_by_gM = {}
        M = min(self.max_rounds, 12)
        # A buyer's latest claim of each kind supersedes its earlier ones; only those enter the likelihood
        # (older claims still count for reliability via bluff / held). Also bounds the cost per update.
        latest = {}
        for claim in self.claims:
            if claim[4] != "bluff":
                latest[claim[1]] = claim
        live_claims = list(latest.values())
        ys = [(min(j, 12), b - a) for j, b in self.bids]
        syy = sum(y * y for _, y in ys)
        rej = [(min(j, 12), x - a) for j, x in self.rejections]
        # Identification fix: a buyer never bids above its own limit (kit bots cap bids at it, harness-enforced LLM
        # buyers too), so R >= its highest bid, less a rounding slack. Off by default to keep V5-V7 reproducible.
        lo_bid = max(0.0, max((y for _, y in ys), default=0.0) - BID_SLACK) if self.bid_lower else 0.0
        for c in self.cells:
            gl = c.gs(self.j0)
            if eff_prior:
                gM = gl[M]
                if gM not in prior_by_gM:
                    prior_by_gM[gM] = effective_segments(base, gM, width)
                prior = prior_by_gM[gM]
            else:
                prior = base
            sgg = sgy = 0.0
            for j, y in ys:
                gj = gl[j]
                sgg += gj * gj
                sgy += gj * y
            hi = width
            for j, dx in rej:
                gj = gl[j]
                if gj > 1e-9:
                    hi = min(hi, (dx + REJECT_SLACK) / gj)
            lo = lo_bid
            n = len(self.bids)
            segs = [(max(lo, s_lo), min(hi, s_hi), d) for s_lo, s_hi, d in prior if min(hi, s_hi) > max(lo, s_lo)]
            if hi <= lo or not segs:
                c.logw, c.mu, c.sd, c.lo, c.hi, c.z = -1e9, 0.0, 0.0, lo, lo + 1e-9, 1.0
                c.segs = [(lo, lo + 1e-9, 1e9)]
                continue
            c.segs = segs
            if sgg > 1e-12:
                mu, sd = sgy / sgg, c.sigma / math.sqrt(sgg)
                rss = max(0.0, syy - mu * mu * sgg)
                c.mu, c.sd = mu, sd
                mass = sum(c._part(s, s[1], sd) for s in segs)          # prior-weighted posterior mass
                logw = (-n * math.log(c.sigma * 2.5066282746) - rss / (2 * c.sigma ** 2)
                        + math.log(sd * 2.5066282746) + math.log(max(mass, 1e-300)))
            else:                                           # nothing informative yet: the prior on [lo, hi]
                mu, sd = 0.0, 0.0
                c.mu, c.sd = mu, sd
                mass = sum(d * (s_hi - s_lo) for s_lo, s_hi, d in segs)
                logw = -n * math.log(c.sigma * 2.5066282746) - syy / (2 * c.sigma ** 2) + math.log(max(mass, 1e-300))
            logw += math.log(P_STUB if c.stub else 1 - P_STUB) + self.log_t[c.T]
            if self.text_type and c.stub and self.stub_signals:
                logw += math.log(STUB_LR) * min(self.stub_signals, STUB_SIGNALS_MAX)
            for j in self.survived:
                if j >= c.T and not c.stub:
                    logw += math.log(SURVIVE_EPS)
            c.lo, c.hi, c.z = segs[0][0], segs[-1][1], 1.0
            for j, tag, b, v, status in live_claims:
                if tag == "final_offer" and b > 0:
                    near = c.cdf(b - a + 0.03) - c.cdf(b - a - 0.03)
                    logw += math.log(1 + 2 * rho * near + 1.5 * rho * (c.T <= j + 2))
                elif tag == "walk_threat":
                    logw += math.log(1 + 1.5 * rho * (c.T <= j + 2 and not c.stub))
                elif tag == "budget_claim" and v is not None:
                    logw += math.log(1 + 2 * rho * (c.cdf(v - a + 0.03) - c.cdf(v - a - 0.03)))
            c.logw = logw
        top = max(c.logw for c in self.cells)
        ws = [math.exp(c.logw - top) for c in self.cells]
        total = sum(ws)
        for c, w in zip(self.cells, ws):
            c.w = w / total

    # ------------------------------------------------------------ outputs
    def willingness(self, cell, R, j):
        a = self.opening
        return a + (R - a) * cell.g(j, self.j0)

    def sample(self, k, seed):
        """k (cell, R) draws: systematic resampling over cells, R by inverse CDF. Deterministic."""
        rng = random.Random(seed)
        u0, out, c, i = rng.random() / k, [], self.cells[0].w, 0
        for m in range(k):
            u = u0 + m / k
            while u > c and i < len(self.cells) - 1:
                i += 1
                c += self.cells[i].w
            cell = self.cells[i]
            out.append((cell, self.opening + cell.inv(rng.random(), self._m(cell))))
        return out

    def p_limit_at_least(self, share, effective=False):
        """P(R >= share); with effective=True, P(willingness at our last round >= share), the limit that matters
        within the game (equal to R for a buyer whose deadline is at or before our last round)."""
        a = self.opening
        if not effective:
            return sum(c.w * (1.0 - c.cdf(share - a, self._m(c))) for c in self.cells)
        M = min(self.max_rounds, 12)
        total = 0.0
        for c in self.cells:
            gM = c.gs(self.j0)[M]
            total += c.w * ((share <= a) if gM <= 1e-9 else (1.0 - c.cdf((share - a) / gM, self._m(c) / gM)))
        return total

    def _m(self, cell):
        """Model-error spread on theta for this cell. "const": MODEL_SD throughout. "remaining" (identification fix):
        scaled by the share of the way to the limit the bids have not covered yet, 1 - g(last bid round), never below
        SD_MIN_SHARE of it: a buyer whose bids have nearly reached its limit leaves little room for model error."""
        if self.sd_mode != "remaining" or self._jl is None:
            return self.model_sd
        return self.model_sd * max(SD_MIN_SHARE, 1.0 - cell.gs(self.j0)[self._jl])

    def p_feasible(self, effective=False):
        return self.p_limit_at_least(self.floor_ratio, effective)

    def limit_estimate(self, lo=0.05, hi=0.95, effective=False):
        """(median, lo-quantile, hi-quantile) of the buyer's limit R (or effective limit), by bisection."""
        a = self.opening

        def q(p):
            x0, x1 = a, 1.0
            for _ in range(20):                         # 2^-20 of list: far finer than any price step
                mid = (x0 + x1) / 2
                if 1.0 - self.p_limit_at_least(mid, effective) < p:
                    x0 = mid
                else:
                    x1 = mid
            return (x0 + x1) / 2
        return q(0.5), q(lo), q(hi)

    def p_leave_by(self, j):
        """Chance the buyer walks after our offer at round j or earlier, if not accepted."""
        return sum(c.w for c in self.cells if not c.stub and c.T <= j)

    def p_accept(self, x, j):
        """Chance an offer at share x of list is accepted at round j."""
        a, total = self.opening, 0.0
        for c in self.cells:
            gj = c.g(j, self.j0)
            if gj <= 1e-9:
                total += c.w * (x <= a)
            else:
                total += c.w * (1.0 - c.cdf((x - a) / gj, self._m(c)))
        return total
