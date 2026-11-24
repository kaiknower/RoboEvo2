"""Eight-condition benchmark protocol; no model or simulator imports."""
from dataclasses import asdict, dataclass
from pathlib import Path

SUITES = ("spatial", "object", "goal", "long")
AXES = {"position": "swap", "task": "task"}
SPLITS = {"train": tuple(range(10)), "validation": tuple(range(20, 30)), "test": tuple(range(10, 20))}
CAPS = {"spatial": 220, "object": 280, "goal": 300, "long": 520}
HF_BASELINE = "YOUR_HF_ORG/libero-pro"


@dataclass(frozen=True)
class Condition:
    suite: str
    axis: str

    def __post_init__(self):
        if self.suite not in SUITES or self.axis not in AXES:
            raise ValueError(f"Unknown condition: {self.suite}/{self.axis}")

    @property
    def name(self):
        return f"{self.suite}_{self.axis}"

    @property
    def benchmark(self):
        base = "10" if self.suite == "long" else self.suite
        return f"libero_{base}_{AXES[self.axis]}"

    @classmethod
    def parse(cls, value):
        return cls(*value.split("_", 1))


def conditions():
    return [Condition(suite, axis) for suite in SUITES for axis in AXES]


def plan(output_root="runs"):
    return {
        "schema_version": 1,
        "method": "PPO + Flow-SDE, initialized from pi05 v8",
        "baseline": HF_BASELINE,
        "splits": SPLITS,
        "training": {"iterations": 200, "validate_every_iteration": True,
                     "validation_episodes_per_condition": 100,
                     "checkpoint_retention": "latest_and_validation_best"},
        "runs": [dict(asdict(c), condition=c.name, benchmark=c.benchmark,
                      tasks=list(range(10)), action_step_cap=CAPS[c.suite],
                      output=str(Path(output_root) / c.name)) for c in conditions()],
        "status": "plan_only_not_trained",
    }
