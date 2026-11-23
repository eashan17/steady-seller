# Seller agent: deterministic ML seller (V11)

Our submission for AgentX Challenge 2, "Selling to Machines". `README.md` is the organisers' starter-kit README and is unchanged. This file describes our seller.

## Run it

```
python evaluate.py --seller seller.py              # official scoring, kit interface (Seller class)
python evaluate.py --seller seller.py --log transcripts.jsonl
python -m unittest discover -s tests               # 119 tests
```

- **Python:** 3.9+, standard library only.
- **Optional:** if `onnxruntime` and `numpy` are installed, the text classifier runs as int8 ONNX. Otherwise a pure-Python copy distilled from it is used automatically. Both score the same on the default run (51.4).
- **Needs nothing else:** no API keys, no network calls, no LLM at runtime.
- **On Windows,** use relative paths and add `--no-baseline`. The kit's loader splits absolute `C:\` paths on the colon. Add `-X utf8` for the ₹ sign.

| Environment variable (all optional) | Effect |
|---|---|
| `SELLER_MODE` | `V0`–`V11`; the default is `V11` (V9 plus a fix so offers never rise). Earlier modes are kept so the ablations can be reproduced |
| `TEXTSIGNALS_BACKEND` | `auto` (the default), `onnx` or `linear` (pure Python) |

## Results (V11)

| Run | Score | Policy / truth violations | Deadlocks |
|---|---:|---:|---:|
| Kit default run (seed 0, 210 negotiations) | **51.4** (baseline 48.5) | 0 / 0 | 0 |
| By kit bot: cooperative / hard bargainer / manipulator | 71.7 / 42.9 / 39.6 (baseline 69.1 / 40.8 / 35.5) | 0 / 0 | 0 |
| Our LLM buyers, Groq gpt-oss-120b, 50 paired negotiations | 43.5 (V5: 44.3, tie) | 0 / 0 | 0 |

The default-run transcripts are in the submission folder (`outputs/transcripts_default_run.jsonl`).

## Layout

| Path | Contents |
|---|---|
| `seller.py` | The `Seller` class: mirrors the harness state, vets every reply through the guard, and falls back safely |
| `negotiator/textsignals.py`, `keywords.py` | 18 text tags (intent, claims, attacks, tone) plus numbers |
| `negotiator/belief.py` | Bayesian grid over the buyer's limit, deadline, concession shape, stubbornness and noise |
| `negotiator/decide.py` | Expected-value planner (offer, accept or walk) |
| `negotiator/templates.py` | Reply phrasing by purpose; numbers only from our own offer |
| `negotiator/guard.py`, `textcheck.py` | Policy and truth checks (the kit's own, plus stricter house rules) |
| `negotiator/fallback.py` | Deterministic rule engine, used whenever anything fails |
| `models/` | ONNX int8 encoder (23 MB) and the pure-Python linear model (4 MB) |
| `training/` | Offline: dataset build, SetFit training, ONNX export, distillation |
| `bench/` | Offline: multi-suite bench, oracle loss breakdown, identification diagnostics, LLM-played buyers |
| `AGENT_CARD.md` | What the seller can and cannot negotiate |
| `demo/` | Demo app: replays, side by side, live "try to break it" (`python -X utf8 demo/server.py`) |

## Our own LLM buyers (stretch goal)

`bench/llm_buyers.py` adds LLM-played buyers in five styles: cooperative, hard bargainer, manipulator, early leaver and impossible budget. Each has a secret limit drawn the kit's way. Backends:

- the Codex CLI
- Groq: set `LLM_BACKEND=groq`, put the key in `GROQ_API_KEY` (read from the environment only, never written to disk), and optionally set `GROQ_MODEL`

```
python bench/run_bench.py --seller seller.py label=v9 mode=V9 --suites llm_mix --seeds 0 --workers 3 --out runs/llm
```
