"""Scoped compatibility for LeRobot 0.1.0 with datasets 4.8.x.

LeRobot 0.1 stacks ``torch.stack(dataset["numeric_column"])``. With a custom
HF transform, datasets 4.x iterates full rows for a Column, needlessly decoding
every image. Numeric Arrow columns are read in one vectorized operation; image
columns retain their original per-row transform path.
"""

from __future__ import annotations

from typing import Any

_ORIGINAL_STACK = None
_ORIGINAL_WORKER_INIT = None


def _column_tensor(column: Any, *, torch: Any, datasets: Any) -> Any | None:
    if not isinstance(column, datasets.Column):
        return None
    source = column.source
    name = column.column_name
    try:
        import pyarrow as pa
        import numpy as np

        array = source._data.column(name)
        indices_table = getattr(source, "_indices", None)
        if indices_table is not None:
            indices = indices_table.column(0)
            array = array.take(indices)
        if hasattr(array, "combine_chunks"):
            array = array.combine_chunks()
        arrow_type = array.type
        if pa.types.is_fixed_size_list(arrow_type):
            value_type = arrow_type.value_type
            if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type) or pa.types.is_boolean(value_type)):
                return None
            values = np.asarray(array.flatten().to_numpy(zero_copy_only=False))
            values = values.reshape((len(array), arrow_type.list_size))
        elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
            value_type = arrow_type.value_type
            if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type) or pa.types.is_boolean(value_type)):
                return None
            offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False))
            lengths = np.diff(offsets)
            if len(lengths) == 0 or np.any(lengths != lengths[0]):
                return None
            values = np.asarray(array.flatten().to_numpy(zero_copy_only=False))
            values = values.reshape((len(array), int(lengths[0])))
        elif pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or pa.types.is_boolean(arrow_type):
            values = np.asarray(array.to_numpy(zero_copy_only=False))
        else:
            return None
        return torch.from_numpy(np.array(values, copy=True, order="C"))
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _install_stack_patch() -> None:
    global _ORIGINAL_STACK
    import datasets
    import torch

    if not hasattr(datasets, "Column"):
        return
    if getattr(torch.stack, "_libero_v8_column_compat", False):
        return
    _ORIGINAL_STACK = torch.stack

    def stack_compat(items: Any, *args: Any, **kwargs: Any):
        vectorized = _column_tensor(items, torch=torch, datasets=datasets)
        if vectorized is not None:
            dim = kwargs.pop("dim", args[0] if args else 0)
            out = kwargs.pop("out", None)
            if kwargs or len(args) > 1:
                return _ORIGINAL_STACK([torch.as_tensor(item) for item in items], *args, **kwargs)
            if dim not in (0, -vectorized.ndim):
                vectorized = _ORIGINAL_STACK(list(torch.unbind(vectorized, dim=0)), dim=dim)
            if out is not None:
                out.copy_(vectorized)
                return out
            return vectorized
        if isinstance(items, datasets.Column):
            items = [item if isinstance(item, torch.Tensor) else torch.as_tensor(item) for item in items]
        return _ORIGINAL_STACK(items, *args, **kwargs)

    stack_compat._libero_v8_column_compat = True  # type: ignore[attr-defined]
    torch.stack = stack_compat


def worker_init_fn(worker_id: int) -> None:
    """Install the Arrow fast path in spawned workers, then preserve OpenPI init."""
    global _ORIGINAL_WORKER_INIT
    _install_stack_patch()
    if _ORIGINAL_WORKER_INIT is None:
        from openpi.training.data_loader import _worker_init_fn as original_worker_init

        original_worker_init(worker_id)
    else:
        _ORIGINAL_WORKER_INIT(worker_id)


def install() -> None:
    """Patch only torch.stack(Column) and register an importable spawn hook."""
    global _ORIGINAL_WORKER_INIT
    _install_stack_patch()
    import openpi.training.data_loader as data_loader

    if data_loader._worker_init_fn is not worker_init_fn:
        _ORIGINAL_WORKER_INIT = data_loader._worker_init_fn
        data_loader._worker_init_fn = worker_init_fn
