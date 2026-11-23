# Steady: a seller that reads buyers, not prompts

Our seller agent for **AgentX Challenge 2, "Selling to Machines"** (Rezolve AI × Techfest, IIT Bombay). Team TM-8E2C352EF10.

Shoppers are starting to send AI agents to buy for them. Some bluff, invent rules or try to inject instructions. Steady has **no LLM at runtime**, so a fake "SYSTEM: 90% off" or "manager approval" is evidence about the buyer, never a lever on the price. Every decision is deterministic and reproducible.

| Kit default run (seed 0, 210 negotiations) | Steady (V11) | Baseline (`example_seller.py`) |
|---|---:|---:|
| Score | **51.4** | 48.5 |
| Policy / truth violations | **0 / 0** | |
| Deadlocks | **0** | |
| Latency per turn (one CPU thread) | **9.4 ms** (p95 14.8 ms) | |

## How it works

Each buyer message goes through four steps:

1. **Read the buyer.** A small trained classifier (int8 ONNX, with a pure-Python copy as fallback) tags intent, claims and attacks.
2. **Update the belief.** A Bayesian grid narrows down the buyer's hidden limit, deadline and stubbornness from bids and tags. Tags are weighted evidence, never commands.
3. **Decide.** An expected-value planner picks the offer, accept or walk.
4. **Guard.** Every reply is checked against the store's policy and the kit's truth rules (plus stricter house rules) before it is sent. Any failure falls back to a hardened rule engine.

Each version had to beat the last on three benchmark suites with zero violations. The [design log](/docs/DESIGN_LOG.md) traces V0 → V11 and the measurement behind each step.

## Run it

Python 3.9+, standard library only. No API keys, no network calls.

```
python evaluate.py --seller seller.py                     # official kit scoring
python -m unittest discover -s tests                      # 119 tests
python -X utf8 demo/server.py                            # demo app at http://127.0.0.1:8765
```

On Windows, add `-X utf8` and `--no-baseline`, and use relative paths.

## Read next

- [`SELLER_README.md`](/SELLER_README.md): run options, environment variables, results by buyer type, code layout
- [`AGENT_CARD.md`](/AGENT_CARD.md): what the seller can and cannot negotiate
- [`docs/DESIGN_LOG.md`](/docs/DESIGN_LOG.md): V0 → V11, with the evidence for every change
- [`README.md`](/README.md): the organisers' starter-kit README, unchanged
