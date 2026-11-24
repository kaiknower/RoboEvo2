#!/usr/bin/env python3
"""Train the joint LIBERO-PRO SFT, CPD and EWC objective with OpenPI JAX."""

import argparse
import dataclasses
import fcntl
import functools
import hashlib
import importlib.util
from itertools import chain
import json
import math
import os
from pathlib import Path
import sys
import time

from recipes import BC_REPO, BC_SOURCE_BATCH, ExposureSchedule, PAIR_REPO, PAIR_SOURCE_BATCH, recipe_for


SOURCE_HASHES = {
    "scripts/train.py": "d5a8b0a24728c0dc9d6127bf756bd29195cb7c705a75dcada9f1b5ac59363b90",
    "src/openpi/training/config.py": "42e4baa2aca12f6ee292c4d52ff76514a1d95e0892e8b0ca709650fe57536734",
    "src/openpi/training/data_loader.py": "5487e52b5d7835ea06905dcc6b577eae1014caf85aa97310603f5c1c3921717a",
    "src/openpi/training/pref_data.py": "4be968ce5789749de8358988e3d9ac75ff4cad8c95d876efd946afcc9dd1bd8d",
    "src/openpi/models/pi0_pref.py": "50dbc368117dbe0484312688c20181442e0ed8604fd9d20e1b6665435234e323",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_fisher(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        sites = {key.removesuffix("::u") for key in keys if key.endswith("::u")}
        if len(sites) != 10 or keys != {f"{site}::{kind}" for site in sites for kind in ("u", "v", "mass")}:
            raise ValueError(f"Expected exactly ten complete Fisher sites at {path}")
        factors = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
    if any(not np.isfinite(value).all() or (value < 0).any() for value in factors.values()):
        raise ValueError(f"Fisher factors must be finite and nonnegative at {path}")
    for site in sites:
        u, v, mass = (factors[f"{site}::{kind}"] for kind in ("u", "v", "mass"))
        if u.ndim < 1 or v.ndim < 1 or u.shape[:-1] != mass.shape or v.shape[:-1] != mass.shape:
            raise ValueError(f"Inconsistent Fisher factor dimensions for {site}")
        if (mass <= 0).any():
            raise ValueError(f"Nonpositive Fisher mass for {site}")
    if not any((value > 0).any() for key, value in factors.items() if key.endswith("::u")):
        raise ValueError(f"All Fisher importance weights are zero at {path}")
    return factors


def load_openpi(root):
    for name, expected in SOURCE_HASHES.items():
        if sha256(root / name) != expected:
            raise ValueError(f"OpenPI source differs from the pinned companion runtime: {name}")
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    from openpi.training import config, weight_loaders

    from cpd_ewc_runtime import patched_trainer_path

    patched = patched_trainer_path(root, SOURCE_HASHES)
    spec = importlib.util.spec_from_file_location("libero_v8_combined_cpd_ewc_train", patched)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return config, weight_loaders, trainer


def dataset_inventory(root, repo, episodes, frames, *, preference=False):
    path = root / repo
    meta = path / "meta"
    files = [meta / name for name in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")]
    if preference:
        files.append(path / "pairs_manifest.json")
    info = json.loads(files[0].read_text())
    count = sum(1 for file in (path / "data").rglob("*.parquet") if file.is_file())
    if (info["total_episodes"], info["total_frames"], count) != (episodes, frames, episodes):
        raise ValueError(f"Unexpected LeRobot v2.1 inventory at {path}")
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 at {path}")
    return {str(file.relative_to(path)): sha256(file) for file in files}


def resolved_config(args, recipe, dev_count):
    config, loaders, trainer = load_openpi(args.openpi_root)
    from openpi.shared import normalize

    source = config.get_config(recipe.source_config)
    norm = args.assets / BC_REPO / "norm_stats.json"
    normalize.load(norm.parent)
    inventory = {BC_REPO: dataset_inventory(args.data_root, BC_REPO, 6593, 1119092)}
    if recipe.cpd_enabled:
        inventory[PAIR_REPO] = dataset_inventory(args.data_root, PAIR_REPO, 701, 701, preference=True)
    if not (args.params / "_METADATA").is_file():
        raise FileNotFoundError(f"Expected an OpenPI Orbax params directory: {args.params}")
    fsdp = args.fsdp_devices
    if fsdp is None:
        fsdp = 4 if recipe.name == "sft-cpd" else 1
    if recipe.phase == "larger-batch" and (fsdp != 4 or (not args.check_only and dev_count != 4)):
        raise ValueError("The larger-batch phase requires four devices with four-device FSDP")
    if not args.check_only:
        widths = [recipe.batch, recipe.pair_batch, fsdp]
        if widths[0] % dev_count or widths[1] % dev_count or dev_count % widths[2]:
            raise ValueError("Training batches must divide evenly over devices and FSDP must divide device count")
    workers = args.num_workers
    if workers is None:
        workers = source.num_workers if recipe.name == "original-sft" else 4
    schedule = source.lr_schedule
    if recipe.schedule_scale != 1.0:
        schedule = ExposureSchedule(schedule, recipe.first_update, recipe.schedule_scale)
    cfg = dataclasses.replace(
        source,
        exp_name="train",
        data=dataclasses.replace(source.data, assets=config.AssetsConfig(assets_dir=str(args.assets), asset_id=BC_REPO)),
        weight_loader=loaders.CheckpointWeightLoader(str(args.params)),
        batch_size=recipe.batch,
        cpd_enabled=recipe.cpd_enabled,
        cpd_pairs_repo_id=PAIR_REPO if recipe.cpd_enabled else None,
        cpd_pref_batch_size=recipe.pair_batch or None,
        cpd_lambda_pref=0.02 if recipe.cpd_enabled else 0.0,
        cpd_lambda_prox=recipe.proximal_weight,
        cpd_margin_enabled=False,
        cpd_pref_use_anchor_reference=False,
        cpd_redirection_enabled=False,
        ewc_enabled=True,
        ewc_lambda=args.ewc_lambda,
        ewc_fisher_path=str(args.fisher),
        ewc_full_diagonal=False,
        lwf_enabled=False,
        vra_enabled=False,
        fsdp_devices=fsdp,
        lr_schedule=schedule,
        num_train_steps=recipe.end_update,
        num_workers=workers,
        save_interval=recipe.saved_every,
        wandb_enabled=False,
        resume=args.resume,
        overwrite=False,
    )
    metadata = {
        "recipe": dataclasses.asdict(recipe),
        "seed": cfg.seed,
        "fsdp_devices": fsdp,
        "num_workers": workers,
        "bc_and_pairs": inventory,
        "normalization_sha256": sha256(norm),
        "initial_params_metadata_sha256": sha256(args.params / "_METADATA"),
        "initial_params_directory": str(args.params),
        "ewc_enabled": True,
        "ewc_lambda": args.ewc_lambda,
        "ewc_fisher_path": str(args.fisher),
        "ewc_fisher_sha256": sha256(args.fisher),
        "ewc_fisher_format": "archived_lora_effective_weight_factorized_npz_v1",
        "ewc_anchor_params": str(args.params),
        "ewc_sites_expected": 10,
        "continuation_checkpoint": str(args.from_checkpoint) if args.from_checkpoint else None,
        "openpi_source_sha256": SOURCE_HASHES,
    }
    return cfg, trainer, metadata


def packed_batches(loader, consumed, source_width, widths):
    import jax
    import numpy as np

    from resume_stream import advance_indices

    def raw_batches():
        while True:
            yield from loader.torch_loader

    full_batches, residual = divmod(consumed, source_width)
    if full_batches:
        advance_indices(loader, full_batches)
    batches = raw_batches()
    if residual:
        batches = chain((jax.tree.map(lambda item: item[residual:], next(batches)),), batches)
    pending = None
    count = 0
    for width in widths:
        chunks = [] if pending is None else [pending]
        while count < width:
            batch = next(batches)
            batch_count = jax.tree.leaves(batch)[0].shape[0]
            if any(item.shape[0] != batch_count for item in jax.tree.leaves(batch)):
                raise ValueError("Collated batch leaves disagree on their sample axis")
            chunks.append(batch)
            count += batch_count
        pending = chunks[0] if len(chunks) == 1 else jax.tree.map(
            lambda *parts: np.concatenate(parts, axis=0), *chunks
        )
        yield jax.tree.map(lambda item: item[:width], pending)
        count -= width
        pending = jax.tree.map(lambda item: item[width:], pending) if count else None


def validate_continuation_source(recipe, metadata):
    source = Path(metadata["continuation_checkpoint"])
    if source.name != str(recipe.first_update - 1) or source.parent.name != "checkpoints":
        raise ValueError(f"Expected a checkpoint directory for committed update {recipe.first_update}: {source}")
    if json.loads((source / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs", 0) <= 0:
        raise ValueError(f"Checkpoint is uncommitted: {source}")
    original = json.loads((source.parent.parent / "resolved_config.json").read_text())
    expected = {
        **metadata,
        "recipe": dataclasses.asdict(recipe_for("sft-cpd")),
        "continuation_checkpoint": None,
    }
    if expected != original:
        different = sorted(key for key in expected.keys() | original.keys() if expected.get(key) != original.get(key))
        raise ValueError(f"Source run configuration differs from the continuation in {different}: {source}")
    if sha256(source / "assets" / BC_REPO / "norm_stats.json") != metadata["normalization_sha256"]:
        raise ValueError(f"Source checkpoint normalization differs from the continuation: {source}")


def restore_checkpoint(source, abstract_state, target_sharding):
    """Read a complete Orbax checkpoint into the current device layout."""
    import orbax.checkpoint as ocp
    from openpi.shared import array_typing
    from openpi.training import checkpoints

    if json.loads((source / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs", 0) <= 0:
        raise ValueError(f"Checkpoint is uncommitted: {source}")
    with array_typing.disable_typechecking(), ocp.PyTreeCheckpointer() as reader:
        state_tree, params_tree = checkpoints._split_params(abstract_state)
        state_sharding, params_sharding = checkpoints._split_params(target_sharding)
        restored_state = reader.restore(
            source / "train_state", item=state_tree,
            restore_args=ocp.checkpoint_utils.construct_restore_args(state_tree, state_sharding),
        )
        params_item = {"params": params_tree}
        restored_params = reader.restore(
            source / "params", item=params_item,
            restore_args=ocp.checkpoint_utils.construct_restore_args(params_item, {"params": params_sharding}),
        )
    return checkpoints._merge_params(restored_state, restored_params)


def run(args, recipe, cfg, trainer):
    import jax
    import numpy as np
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.models import model
    from openpi.training import checkpoints, data_loader, sharding
    from lerobot_datasets_compat import install

    install()
    trainer.init_logging()
    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    jax.config.update("jax_compilation_cache_dir", str(cache / "jax"))
    train_rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    source_cfg = dataclasses.replace(cfg, batch_size=BC_SOURCE_BATCH, cpd_pref_batch_size=PAIR_SOURCE_BATCH)
    bc_loader = data_loader.create_data_loader(source_cfg, sharding=data_sharding, shuffle=True)
    pair_loader = None
    if recipe.cpd_enabled:
        from openpi.training import pref_data

        pair_loader = pref_data.create_pref_loader(source_cfg, sharding=data_sharding, shuffle=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2 if recipe.name == "sft-only" else 1,
        keep_period=cfg.keep_period if recipe.name == "original-sft" else None,
        create=False,
        async_options=ocp.AsyncOptions(timeout_secs=7200),
    )
    handlers = {
        "assets": checkpoints.CallbackHandler(),
        "train_state": ocp.PyTreeCheckpointHandler(),
        "params": ocp.PyTreeCheckpointHandler(),
    }
    restore = args.resume or args.from_checkpoint is not None
    with ocp.CheckpointManager(checkpoint_dir, item_handlers=handlers, options=options) as manager:
        state, state_sharding = trainer.init_train_state(cfg, init_rng, mesh, resume=restore)
        if recipe.name != "original-sft":
            leaves = list(state.params.filter(cfg.trainable_filter).flat_state().items())
            count = sum(math.prod(value.value.shape) for _, value in leaves)
            if len(leaves) != 20 or count != 1_245_708_288 or any(
                not any("lora" in str(segment).lower() for segment in key) for key, _ in leaves
            ):
                raise RuntimeError("Merged-v8 continuation requires the 20 rank-512 LoRA leaves")
            del leaves
        expected_start = recipe.first_update
        source = args.from_checkpoint
        if args.resume:
            last = manager.latest_step()
            if last is None:
                raise FileNotFoundError(f"No committed checkpoint in {checkpoint_dir}")
            source = checkpoint_dir / str(last)
            expected_start = last + 1
        if restore:
            state = restore_checkpoint(source, state, state_sharding)
        start = int(state.step)
        if start != expected_start or start > recipe.end_update:
            raise ValueError(f"Checkpoint optimizer step {start} does not match expected step {expected_start}")
        if restore:
            counters = [
                int(value) for value in jax.tree.leaves(state.opt_state)
                if hasattr(value, "shape") and value.shape == () and np.issubdtype(value.dtype, np.integer)
            ]
            if not counters or any(count != start for count in counters):
                raise ValueError(f"Checkpoint optimizer counters do not match step {start}: {counters}")
        stop = recipe.end_update if args.probe_updates is None else start + args.probe_updates
        if args.stop_after is not None:
            stop = args.stop_after
        if stop < start:
            raise ValueError(f"Requested stop at update {stop} precedes the restored update {start}")
        if start == stop:
            print(f"Checkpoint already reached requested update {start}")
            return
        steps = range(start, stop)
        bc_batches = packed_batches(
            bc_loader, recipe.bc_examples(start), BC_SOURCE_BATCH, (recipe.batch_at(step)[0] for step in steps),
        )
        pairs = None
        in_shardings = (replicated, state_sharding, data_sharding)
        if pair_loader is not None:
            pairs = packed_batches(
                pair_loader, recipe.pair_examples(start), PAIR_SOURCE_BATCH, (recipe.batch_at(step)[1] for step in steps),
            )
            in_shardings += (data_sharding,)
        ewc_fisher = {key: jnp.asarray(value) for key, value in load_fisher(args.fisher).items()}
        kernel = jax.jit(
            functools.partial(trainer.train_step, cfg, ewc_fisher=ewc_fisher),
            in_shardings=in_shardings,
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )

        def on_devices(value):
            return jax.tree.map(lambda item: jax.make_array_from_process_local_data(data_sharding, item), value)

        with (args.output / f"metrics-{start}-{time.time_ns()}.jsonl").open("x", buffering=1) as log:
            for step in steps:
                loaded_at = time.perf_counter()
                raw = next(bc_batches)
                batch = on_devices((model.Observation.from_dict(raw), raw["actions"]))
                preference = () if pairs is None else (on_devices(next(pairs)),)
                step_started = time.perf_counter()
                with sharding.set_mesh(mesh):
                    state, metrics = kernel(train_rng, state, batch, *preference)
                metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"Nonfinite metrics at step {step}: {metrics}")
                if metrics.get("ewc_sites") != 10:
                    raise RuntimeError(f"Combined CPD+EWC expected 10 active EWC sites: {metrics}")
                if pairs is not None:
                    expected_loss = (
                        metrics["loss_bc"]
                        + cfg.cpd_lambda_pref * metrics["loss_pref"]
                        + 0.5 * cfg.ewc_lambda * metrics["ewc_penalty"]
                    )
                    if not math.isclose(metrics["loss"], expected_loss, rel_tol=2e-6, abs_tol=1e-8):
                        raise RuntimeError(f"Combined objective mismatch at step {step}: {metrics}")
                if step > 0 and metrics.get("ewc_penalty", 0.0) <= 0.0:
                    raise RuntimeError(f"EWC penalty remained inactive after optimizer updates: {metrics}")
                if step == 0 and metrics["grad_norm"] <= 0:
                    raise FloatingPointError("The first update has zero gradient")
                if pairs is None and set(metrics) != {"loss", "grad_norm", "param_norm"}:
                    raise RuntimeError(f"SFT used an unexpected objective: {metrics}")
                if recipe.name == "sft-cpd-no-prox" and metrics["loss_pref"] != metrics["cpd_l_dpo"]:
                    raise RuntimeError("No-prox preference loss does not match the IPO term")
                update = step + 1
                record = {
                    "step": step, "updates": update, "bc_examples": recipe.bc_examples(update),
                    "pairs": recipe.pair_examples(update), **metrics,
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if args.probe_updates is not None:
                    widths = recipe.batch_at(step)
                    jax.block_until_ready(state)
                    step_seconds = time.perf_counter() - step_started
                    memory_keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "bytes_reserved")
                    memory = [{key: stats[key] for key in memory_keys if key in stats}
                              for device in jax.local_devices() if (stats := device.memory_stats()) is not None]
                    print(json.dumps(dict(probe_update=update, bc_batch=widths[0], pair_batch=widths[1],
                                          load_seconds=step_started - loaded_at,
                                          first_at_width=step == start or widths != recipe.batch_at(step - 1),
                                          step_seconds=step_seconds, memory_bytes=memory)), flush=True)
                    continue
                if step % cfg.log_interval == 0 or update == stop:
                    print(f"Update {update}/{recipe.end_update} loss={metrics['loss']:.6f}", flush=True)
                save_first = step == recipe.first_update
                periodic = update % recipe.saved_every == 0
                if save_first or periodic or update == stop:
                    checkpoints.save_state(manager, state, bc_loader, step)
            manager.wait_until_finished()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("sft-cpd",))
    parser.add_argument("--phase", choices=("full", "larger-batch"), default="full")
    parser.add_argument("--batch-size", type=int, help="New global BC width for the larger-batch phase")
    parser.add_argument("--first-update", type=int, help="Completed optimizer updates at the larger-batch handoff")
    ending = parser.add_mutually_exclusive_group()
    ending.add_argument("--probe-updates", type=int, help="Benchmark this many updates without saving checkpoints")
    ending.add_argument("--stop-after", type=int, help="Save and stop at this absolute optimizer update")
    parser.add_argument("--openpi-root", type=Path, help="vendor/openpi in the pinned LIBERO-PRO runtime")
    parser.add_argument("--params", type=Path, help="OpenPI JAX initialization params directory")
    parser.add_argument("--assets", type=Path, help=f"Directory containing {BC_REPO}/norm_stats.json")
    parser.add_argument("--data-root", type=Path, help="Local LeRobot v2.1 root containing dataset repo paths")
    parser.add_argument("--output", type=Path, help="Output directory for metrics and committed checkpoints")
    parser.add_argument("--fisher", type=Path,
                        help="Factorized effective-weight LoRA Fisher NPZ computed for --params")
    parser.add_argument("--ewc-lambda", type=float, default=4e-6)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--fsdp-devices", type=int)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--from-checkpoint", type=Path, help="Committed source checkpoint for the larger-batch handoff")
    restore.add_argument("--resume", action="store_true", help="Restore this run's latest committed checkpoint")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print the recipe without loading dependencies or data")
    mode.add_argument("--check-only", action="store_true", help="Verify source, inputs and config without training")
    args = parser.parse_args()
    try:
        recipe = recipe_for(args.recipe, args.phase, args.batch_size, first_update=args.first_update)
    except ValueError as error:
        parser.error(str(error))
    if args.stop_after is not None and not recipe.first_update < args.stop_after <= recipe.end_update:
        parser.error("--stop-after must be after the phase's first update and within its budget")
    if not math.isclose(args.ewc_lambda, 4e-6, rel_tol=0.0, abs_tol=1e-15):
        parser.error("The combined recipe uses --ewc-lambda 4e-6")
    plan = {
        **dataclasses.asdict(recipe), "cpd_enabled": recipe.cpd_enabled,
        "sft_examples_at_end": recipe.bc_examples(recipe.end_update),
        "preference_pairs_at_end": recipe.pair_examples(recipe.end_update),
        "final_checkpoint_index": recipe.end_update - 1,
        "ewc_enabled": True,
        "ewc_lambda": args.ewc_lambda,
        "ewc_fisher_format": "archived_lora_effective_weight_factorized_npz_v1",
    }
    if args.plan:
        if args.stop_after is not None:
            plan["stop_after"] = args.stop_after
        print(json.dumps(plan, indent=2))
        return
    required = ["openpi_root", "params", "assets", "data_root", "fisher"]
    if not args.check_only:
        required.append("output")
    for attr in [*required, "from_checkpoint"]:
        value = getattr(args, attr)
        if value is None and attr in required:
            parser.error(f"--{attr.replace('_', '-')} is required")
        if value is not None:
            setattr(args, attr, value.expanduser().resolve())
    if args.fsdp_devices is not None and args.fsdp_devices < 1:
        parser.error("--fsdp-devices must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.from_checkpoint is not None and recipe.phase != "larger-batch":
        parser.error("--from-checkpoint applies to the larger-batch phase")
    if recipe.phase == "larger-batch" and not args.resume and args.from_checkpoint is None:
        parser.error("The larger-batch phase needs --from-checkpoint or --resume")
    if args.check_only and args.resume:
        parser.error("Use --from-checkpoint to preflight the larger-batch phase")
    if args.probe_updates is not None and (
        recipe.phase != "larger-batch" or args.resume or args.check_only
        or not 0 < args.probe_updates <= recipe.end_update - recipe.first_update
    ):
        parser.error("A probe requires a fresh larger-batch run and a positive update count within its remaining budget")
    os.environ["HF_LEROBOT_HOME"] = str(args.data_root)
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.check_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    if not args.check_only and jax.default_backend() != "gpu":
        parser.error("Training requires a GPU JAX backend. Use --check-only on CPU.")
    cfg, trainer, metadata = resolved_config(args, recipe, jax.device_count())
    if args.check_only:
        if recipe.phase == "larger-batch":
            validate_continuation_source(recipe, metadata)
        load_fisher(args.fisher)
        print(json.dumps({"plan": plan, "seed": cfg.seed, "action_horizon": cfg.model.action_horizon,
                          "fsdp_devices": cfg.fsdp_devices, "source_inputs": metadata}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output / "resolved_config.json"
        if args.resume:
            prior = json.loads(manifest.read_text())
            if args.from_checkpoint is None:
                metadata["continuation_checkpoint"] = prior["continuation_checkpoint"]
            if prior != metadata:
                raise ValueError("Resume input differs from the existing training manifest")
            if recipe.phase == "larger-batch":
                validate_continuation_source(recipe, metadata)
        else:
            if any(path != args.output / ".train.lock" for path in args.output.iterdir()):
                raise FileExistsError(f"Output directory already contains a run: {args.output}")
            if recipe.phase == "larger-batch":
                validate_continuation_source(recipe, metadata)
            with manifest.open("x") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
        run(args, recipe, cfg, trainer)


if __name__ == "__main__":
    main()
