"""LIBERO-PRO v8 training budgets and learning-rate clocks."""

from dataclasses import dataclass

BC_REPO = "roboevo2/libero_expert_v8"
PAIR_REPO = "roboevo2/libero_cpd_strong_pairs_v8champ"
ORIGINAL_CONFIG = "pi05_libero_v8_lora"
CONTINUATION_CONFIG = "pi05_libero_v8_merged_cpd_r512_balanced_lora"
BC_SOURCE_BATCH = 32
PAIR_SOURCE_BATCH = 8
FULL_UPDATES = 30_000
CPD_TRANSITION = 14_001
NAMES = ("original-sft", "sft-only", "sft-cpd", "sft-cpd-no-prox")


@dataclass(frozen=True)
class Recipe:
    name: str
    phase: str
    source_config: str
    batch: int
    pair_batch: int
    proximal_weight: float
    first_update: int
    end_update: int
    schedule_scale: float
    saved_every: int

    @property
    def cpd_enabled(self):
        return self.pair_batch > 0

    def examples(self, update, source_width, batch_width):
        if self.phase == "larger-batch":
            before = min(update, self.first_update)
            count = before * source_width + (update - before) * batch_width
            return min(count, FULL_UPDATES * source_width)
        return self.first_update * source_width + (update - self.first_update) * batch_width

    def bc_examples(self, update):
        return self.examples(update, BC_SOURCE_BATCH, self.batch)

    def pair_examples(self, update):
        return self.examples(update, PAIR_SOURCE_BATCH, self.pair_batch) if self.cpd_enabled else 0

    def batch_at(self, step):
        if not 0 <= step < self.end_update:
            raise ValueError(f"Step outside training recipe: {step}")
        return (
            self.bc_examples(step + 1) - self.bc_examples(step),
            self.pair_examples(step + 1) - self.pair_examples(step),
        )


def recipe_for(name, phase="full", batch=None, *, first_update=None):
    if name not in NAMES:
        raise ValueError(f"Unknown recipe {name!r}")
    if phase not in ("full", "initial", "continuation", "larger-batch"):
        raise ValueError(f"Unknown phase {phase!r}")
    if phase != "full" and name != "sft-cpd":
        raise ValueError("The initial and continuation phases are specific to sft-cpd")
    if phase == "larger-batch":
        if type(first_update) is not int or not 0 < first_update < FULL_UPDATES:
            raise ValueError("The larger-batch phase needs an intermediate --first-update")
        if type(batch) is not int or batch <= BC_SOURCE_BATCH or batch % 16:
            raise ValueError("The larger-batch BC width must be a multiple of 16 and greater than 32")
        remaining = (FULL_UPDATES - first_update) * BC_SOURCE_BATCH
        end_update = first_update + (remaining + batch - 1) // batch
        return Recipe(name, phase, CONTINUATION_CONFIG, batch, batch // 4, 0.5,
                      first_update, end_update, batch / BC_SOURCE_BATCH, 400)
    if first_update is not None:
        raise ValueError("--first-update applies to the larger-batch phase")
    if batch is not None and name != "sft-only":
        raise ValueError("--batch-size applies to sft-only")
    if name == "original-sft":
        return Recipe(name, phase, ORIGINAL_CONFIG, 32, 0, 0.0, 0, 30_000, 1.0, 1_000)
    if name == "sft-only":
        batch = 96 if batch is None else batch
        if batch not in (80, 96):
            raise ValueError("The SFT-only continuation implements batches 80 and 96")
        return Recipe(name, phase, CONTINUATION_CONFIG, batch, 0, 0.0, 0, 960_000 // batch, batch / 32, 2_000)
    if name == "sft-cpd-no-prox":
        return Recipe(name, phase, CONTINUATION_CONFIG, 48, 12, 0.0, 0, 20_000, 1.5, 2_000)
    if phase == "continuation":
        return Recipe(name, phase, CONTINUATION_CONFIG, 48, 12, 0.5, CPD_TRANSITION, 24_667, 1.5, 2_000)
    end = CPD_TRANSITION if phase == "initial" else FULL_UPDATES
    return Recipe(name, phase, CONTINUATION_CONFIG, 32, 8, 0.5, 0, end, 1.0, 2_000)


@dataclass(frozen=True)
class ExposureSchedule:
    base: object
    first_update: int
    scale: float

    def create(self):
        original = self.base.create()

        def schedule(count):
            return original(self.first_update + (count - self.first_update) * self.scale)

        return schedule
