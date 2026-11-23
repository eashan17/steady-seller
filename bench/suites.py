"""Buyer suites for the S2 bench, and the registry later slices add to.

A suite is a set of `Buyer` subclasses plus the seeds and scenario count to play them on.
Scenarios come from the kit's `buyers.make_scenarios(store, policy, n, seed, classes)`, so the
class attributes (`feasible`, `zone`, `anchor`, `deadline`, `exponent`, `stubborn`) and the dict
order fix every scenario. The built-in suites copy `tradeoff-analysis.md` §1 exactly:

    public          the kit's three bots                                  seeds 0-4 x 210
    proxy           the MVP's rule-based "LLM-style" buyers               seeds 0-2 x 200
    shifted         public bot types with changed curves and acceptance   seeds 0-2 x 210
    jitter_0.015    proxy buyers with price_jitter = 0.015                seeds 0-2 x 200
    jitter_0.05     ... 0.05
    jitter_0.08     ... 0.08

Adding a suite (S6 LLM buyers, or anything else):

    import suites
    suites.register("llm_coop", {"llm_coop": LLMCooperative}, seeds=[0], n=40,
                    kind="llm", default=False, description="Codex-played cooperative buyer")

`classes` may also be a zero-argument callable returning the dict, so expensive set-up (an LLM
client check, say) runs only when the suite is selected. Buyers must subclass the kit's `Buyer`
and keep its constructor `(scenario, store, rng)`. A module named in `PLUGIN_MODULES` that sits
next to this file is imported by `run_bench.py` automatically; any other module can be loaded
with `run_bench.py --suite-module path/to/module.py`.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import buyers
import proxy_buyers

BuyerClasses = Dict[str, type]

# Modules beside this file that register suites when imported (S6 adds bench/llm_buyers.py).
PLUGIN_MODULES = ("llm_buyers",)


@dataclass
class Suite:
    name: str
    classes: Union[BuyerClasses, Callable[[], BuyerClasses]]
    seeds: List[int]
    n: int
    kind: str = "rule"            # "rule" (deterministic bots) or "llm" (slow, quota-limited)
    default: bool = True          # part of the default run when no --suites is given
    description: str = ""
    _resolved: BuyerClasses = field(default=None, repr=False)

    def buyer_classes(self) -> BuyerClasses:
        if self._resolved is None:
            classes = self.classes() if callable(self.classes) else self.classes
            if not isinstance(classes, dict) or not classes:
                raise ValueError(f"suite {self.name!r}: classes must be a non-empty dict of Buyer subclasses")
            for key, cls in classes.items():
                if not (isinstance(cls, type) and issubclass(cls, buyers.Buyer)):
                    raise TypeError(f"suite {self.name!r}: {key!r} is not a buyers.Buyer subclass")
            self._resolved = dict(classes)
        return self._resolved


REGISTRY: Dict[str, Suite] = {}


def register(name, classes, seeds, n, kind="rule", default=None, description="", replace=False) -> Suite:
    """Add a suite. `default` falls back to True for rule suites and False for LLM suites."""
    if name in REGISTRY and not replace:
        raise ValueError(f"suite {name!r} is already registered (pass replace=True to override)")
    if not seeds or n < 1:
        raise ValueError(f"suite {name!r}: needs at least one seed and n >= 1")
    suite = Suite(name, classes, list(seeds), int(n), kind,
                  (kind == "rule") if default is None else bool(default), description)
    REGISTRY[name] = suite
    return suite


def get(name) -> Suite:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; registered: {', '.join(REGISTRY)}") from None


def default_names() -> List[str]:
    return [s.name for s in REGISTRY.values() if s.default]


# ---------------------------------------------------------------- shifted buyer types
# Copied from context/experiments/tradeoffs.py:182-217 (attributes must stay identical).
class FastCoop(buyers.Cooperative):
    deadline = 4


class PatientCoop(buyers.Cooperative):
    deadline, stubborn = 10, 0.0


class LinearHard(buyers.HardBargainer):
    exponent = 1.0


class LateStubborn(buyers.HardBargainer):
    exponent, stubborn = 4.0, 1.0


class EarlyQuit(buyers.HardBargainer):
    deadline, stubborn = 6, 0.0


class Threshold(buyers.Buyer):
    """Accepts any offer at or under its private limit in any round (an 'honest budget' LLM)."""
    feasible, zone, anchor = 0.85, (0.1, 0.9), (0.55, 0.7)
    deadline, exponent, stubborn = 10, 1.0, 0.5

    def respond(self, msg):
        r = msg["round"]
        offered = self.usable(msg.get("items")) if msg.get("type") == "offer" else None
        if offered:
            self.basket = [{"sku": i["sku"], "qty": i["qty"]} for i in offered]
            if self.total(offered) <= self.s["reservation"] * self.listed(offered) + 1:
                return {"type": "accept", "text": "Deal."}
        if r >= self.s["deadline"] and not self.s["stubborn"]:
            return {"type": "walk_away", "text": "No."}
        bid = self.priced(self.basket, self.ratio(r))
        return {"type": "counter", "items": bid, "text": "Counter."}


SHIFTED = {"fast_coop": FastCoop, "patient_coop": PatientCoop, "linear_hard": LinearHard,
           "late_stubborn": LateStubborn, "early_quit": EarlyQuit, "threshold": Threshold}


def jittered(jitter) -> BuyerClasses:
    """The four proxy classes with `price_jitter` overridden (context/experiments/jitter.py:9-10)."""
    return {name: type(f"{cls.__name__}J", (cls,), {"price_jitter": jitter})
            for name, cls in proxy_buyers.BUYERS.items()}


register("public", buyers.BUYERS, [0, 1, 2, 3, 4], 210, description="the kit's three bots")
register("proxy", proxy_buyers.BUYERS, [0, 1, 2], 200, description="MVP rule-based LLM-style buyers")
register("shifted", SHIFTED, [0, 1, 2], 210, description="public types with changed deadlines, curves, acceptance")
for _j in (0.015, 0.05, 0.08):
    register(f"jitter_{_j}", jittered(_j), [0, 1, 2], 200, description=f"proxy buyers, bid noise +/-{_j:.1%}")

# The suites D8 averages over (plan.md D8); noise suites only gate "no drop > 0.2".
D8_MEAN_SUITES = ("public", "proxy", "shifted")
