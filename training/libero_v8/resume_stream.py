"""Restore the seeded batch-index streams without decoding discarded images.

Dataset transforms are deterministic. Model-side augmentation uses the original
seed folded with the restored update count. Worker RNG state is not saved, so a
resumed run is not bitwise identical to an uninterrupted one.
"""
from __future__ import annotations

import itertools


class OffsetBatchSampler:
    def __init__(self, original, skip):
        self.original, self.skip = original, skip

    def __len__(self):
        return len(self.original)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        yield from itertools.islice(iter(self.original), skip, None)


def advance_indices(loader, batches):
    """Match TorchDataLoader epoch/base-seed consumption, then skip indices.

    Supports the original RandomSampler and CPD WeightedRandomSampler. The
    original persistent-worker SFT stream has not completed its first epoch at
    this resume point; reject unimplemented worker-reset cases explicitly.
    """
    import torch

    dl = loader.torch_loader
    full_epochs, offset = divmod(batches, len(dl))
    if dl.persistent_workers and full_epochs:
        raise ValueError("Persistent-worker resume after a full SFT epoch needs a separate RNG restoration path")
    for _ in range(full_epochs):
        torch.empty((), dtype=torch.int64).random_(generator=dl.generator)
        for _batch in dl.batch_sampler:
            pass
    original = dl.batch_sampler
    object.__setattr__(dl, "batch_sampler", OffsetBatchSampler(original, offset))
    return dict(consumed_batches=batches, completed_epochs=full_epochs,
                within_epoch_batches=offset, batches_per_epoch=len(original),
                sampler=type(dl.sampler).__name__, workers=dl.num_workers,
                worker_rng_checkpointed=False, bitwise_equivalence_claimed=False)
