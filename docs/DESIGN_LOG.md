# Design log: V0 → V11

Each version was kept only if it passed one selection rule, on paired seeds:
- the mean over the public, proxy and shifted suites goes up
- no rule suite drops by more than 0.2 points
- policy violations, truth violations and exceptions all stay at zero

Comparisons against LLM buyers are paired, with 95% confidence intervals. Scores are per 100 negotiations.

## Constraints we set

- **No LLM at runtime.** An LLM seller is slow, costs money per call, varies from run to run, and any word it writes is checked by the truth rules. It is also a target for prompt injection.
- **Text is read by small trained models.** The hidden buyers are LLMs, so their intent and tactics live in the text.
- **LLMs are used offline only:** to write labelled training text, and to play test buyers.

## Versions

| Step | What changed | Why (the measurement that led to it) | Result |
|---|---|---|---|
| V0 | Hardened rule seller: fixes the MVP crash, reads buyer messages leniently but accepts strictly, guard on every message | Baseline behaviour, plus probes of the kit's checker | Bench 48.08 / 33.45 / 52.48 (public / proxy / shifted); kit 51.2 |
| V1 | Bayesian buyer model (exact grid) and expected-value planner | Analysis of deadlocks and their cost: deadlocks are cheap insurance only when the buyer can't leave | Every suite better (+0.11 to +1.11); deadlocks down 80–100% |
| V2 / V3 | Text tags from keyword rules (V2) or from the trained classifier (V3) | The hidden buyers write free text | On kit bots no difference: their text is scripted, so these bots can't show what text is worth |
| S6 | Built LLM-played buyers (Codex CLI) and ran 250 negotiations | The kit bots can't test text | Zero violations everywhere; the belief underprices LLM buyers; on real LLM messages the classifier read intent worse than keywords (F1 0.47 vs 0.56; claims 0.31 vs 0.57) |
| V4 / V5 | A fallback price floor, urgency read as leave risk, and hybrid tags (keywords for intent and claims, the classifier for tone and attacks) | Both S6 findings | Kit 51.4; best LLM-buyer point estimate (38.9) |
| Diagnosis | Losses broken down against an oracle (the best play possible given the buyer's true limit and deadline) | Find where the points go | Main losses: accepting bids below what the buyer would pay, and underpricing near the end against buyers who concede late |
| V6 | Direct fix of acceptance: accept guard, bid-gap model, counters anchored to bids | The diagnosis | **Rejected.** The losses moved: buyers left, or we walked. Lesson: the cause lies upstream, in how well the buyer is identified |
| V7–V9 | Identification fixes: limit prior anchored to our floor; limit ≥ the buyer's highest bid; manipulation text as evidence of stubbornness; model error shrinking as bids near the limit | Measured belief accuracy: V5's limit bias was −0.10 to −0.38 of list for late concessors, and stubbornness stayed at a 50/50 prior | Limit bias about ±0.03; stubbornness Brier 0.25 → 0.01; public +0.18 (CI > 0), proxy +0.65, jitter ±5% +0.90, jitter ±8% −1.84 |
| V10 and others | Closed-loop final turn, probe offers, finer grid, quantile caps, "hold then drop" paths, priors on late deadlines | Is anything downstream still fixable? | None passes the rule. Counterfactuals: the remaining kit-bot gap is all margin lost to early commitments, made before the bids reveal the deadline or limit. That is an information limit |
| Final check | 50 paired negotiations against Groq gpt-oss-120b LLM buyers, V9 vs V5 | Choose between the two best versions | V9 − V5 = −0.72, 95% CI [−1.96, +0.51]: a tie, with the same deals, zero violations and zero deadlocks. **V9 chosen:** it identifies buyers better, wins on the kit bots, and ties on LLM buyers |

| V11 | Bug fix: when a planned price was rounded up to whole rupees, it could land a fraction above our previous ask; the old check then sent the fallback schedule's much higher offer, so our price rose mid-negotiation (24 of 1,310 kit offers; 4 of 50 LLM games) | Found while building the demo replays; contradicts the design ("asks never rise") | Repeats the previous offer instead. Rule bench vs V9: public +0.07 (CI > 0), proxy −0.07, shifted −0.03, jitter ±5% −0.23, jitter ±8% −0.34; the accidental rises had been earning a little against noisy bots. Fails D8 narrowly; **chosen by the user for consistency and defensibility**. Kit 51.4, zero violations |

| V11 wording | `honest_wording`: reply wording follows what the price actually did (first offer / lowered / held) | A transcript review found templates chosen from tags alone: "I've moved on price" at an unchanged or list price, "my offer stands" after a drop (205 of 1,310 kit offers) | Wording only: 0 of 210 kit negotiations change price or action; mismatches 205 → 0 |
| Bench fix | The LLM test buyer's structured bid now matches the total it writes (whole rupees, not ₹10 steps) | A replay showed "₹3,099, final" sent as a ₹3,080 bid (text and bid differed in 55 of 120 messages) | Applies to future LLM runs; recorded runs keep their data (equal for every version compared) |

## Known limits

- **±8% bid noise.** It hides whether a deal is possible from every version tried. V9 deadlocks more than V5 against noisy buyers with no feasible deal.
- **Cost of the higher LLM point estimate.** V5's slightly higher estimate against LLM buyers comes from a later concession at round 2. Building that in would need a model of strategic LLM bidding fitted on far more than 50 games. Learning from buyers across negotiations would also do it, but would break the fairness guarantee.
