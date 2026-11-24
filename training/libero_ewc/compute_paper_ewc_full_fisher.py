#!/usr/bin/env python3
"""Compute the exact full-parameter diagonal empirical Fisher for EWC.

This is separate from the factorized LoRA Fisher code. It implements the Kirkpatrick et al. contract for the dense pi0.5 policy:

  F_i = (1 / N) sum_n (d L_A(x_n; theta_A*) / d theta_i) ** 2

Each optimizer-free step has batch size one, so this is the mean of per-example
squared gradients, not the square of a batch-mean gradient.  Every dense model
parameter is included.  The result is saved as an Orbax pytree with exactly the
same key/shape tree as theta_A*; no rank-1 or row/column approximation is made.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import platform

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as _sharding


def _tree_shape(state: nnx.State, dtype=None):
    return state.map(
        lambda _path, leaf: leaf.replace(
            jax.ShapeDtypeStruct(leaf.value.shape, dtype or leaf.value.dtype)
        )
    )


def _init_dense_state(config: _config.TrainConfig, rng, mesh):
    abstract_model = nnx.eval_shape(config.model.create, rng)
    graphdef, state_shape = nnx.split(abstract_model)
    state_sharding = _sharding.fsdp_sharding(state_shape, mesh, log=True)
    partial = config.weight_loader.load(state_shape.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def init(partial_params):
        model = config.model.create(rng)
        _, state = nnx.split(model)
        state.replace_by_pure_dict(partial_params)
        return state

    state = jax.jit(
        init,
        in_shardings=(replicated,),
        out_shardings=state_sharding,
    )(partial)
    jax.block_until_ready(state)
    return graphdef, state, state_sharding


def compute(config: _config.TrainConfig, out: Path, steps: int, seed: int) -> None:
    if not config.ewc_enabled or not config.ewc_full_diagonal:
        raise SystemExit("config is not a dense full-diagonal EWC config")
    if config.fsdp_devices != jax.device_count():
        raise SystemExit(
            f"expected exactly {config.fsdp_devices} visible devices, got {jax.device_count()}"
        )

    if config.batch_size != 1 or config.data.base_config.lerobot_episode_filter_path is not None:
        raise ValueError("expected one task-A example per step and no task-B episode filter")

    mesh = _sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_sharding.BATCH_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(seed)
    init_rng, fisher_rng = jax.random.split(rng)
    graphdef, state, state_sharding = _init_dense_state(config, init_rng, mesh)

    all_paths = set(traverse_util.flatten_dict(state.to_pure_dict(), sep="/"))
    trainable_paths = set(
        traverse_util.flatten_dict(
            state.filter(config.trainable_filter).to_pure_dict(), sep="/"
        )
    )
    if all_paths != trainable_paths:
        raise SystemExit(
            f"full Fisher requires every parameter: {len(trainable_paths)}/{len(all_paths)}"
        )
    if any("lora" in path for path in all_paths):
        raise SystemExit("dense Fisher unexpectedly found LoRA parameters")

    fisher = state.map(
        lambda _path, leaf: leaf.replace(jnp.zeros(leaf.value.shape, jnp.float32))
    )
    fisher_sharding = _sharding.fsdp_sharding(_tree_shape(fisher), mesh)
    fisher = jax.device_put(fisher, fisher_sharding)

    loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        num_batches=steps,
    )

    def example_loss(model, loss_rng, observation, actions):
        return jnp.mean(model.compute_loss(loss_rng, observation, actions, train=False))

    diff_state = nnx.DiffState(0, config.trainable_filter)

    def fisher_step(params, accumulator, loss_rng, observation, actions):
        model = nnx.merge(graphdef, params)
        grads = nnx.grad(example_loss, argnums=diff_state)(
            model, loss_rng, observation, actions
        )
        grad_by_path = dict(grads.flat_state())
        return accumulator.map(
            lambda path, leaf: leaf.replace(
                leaf.value
                + jnp.square(grad_by_path[path].value.astype(jnp.float32))
            )
        )

    pstep = jax.jit(
        fisher_step,
        in_shardings=(state_sharding, fisher_sharding, replicated, data_sharding, data_sharding),
        out_shardings=fisher_sharding,
        donate_argnums=(1,),
    )

    for index, (observation, actions) in enumerate(loader):
        step_rng = jax.random.fold_in(fisher_rng, index)
        fisher = pstep(state, fisher, step_rng, observation, actions)
        if (index + 1) % 16 == 0 or index + 1 == steps:
            jax.block_until_ready(fisher)
            logging.info("exact per-example Fisher %d/%d", index + 1, steps)

    fisher = jax.jit(
        lambda tree: tree.map(
            lambda _path, leaf: leaf.replace(leaf.value / np.float32(steps))
        ),
        in_shardings=(fisher_sharding,),
        out_shardings=fisher_sharding,
        donate_argnums=(0,),
    )(fisher)
    jax.block_until_ready(fisher)

    values = [leaf.value for leaf in fisher.flat_state().values()]
    total_mass = jax.device_get(sum((jnp.sum(x) for x in values), jnp.zeros(())))
    total_params = sum(int(x.size) for x in values)
    nonzero_by_tensor = jax.device_get(tuple(jnp.count_nonzero(x) for x in values))
    nonzero = sum(int(count) for count in nonzero_by_tensor)

    out.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(out, {"params": fisher})

    checkpoint = Path(config.weight_loader.params_path)
    metadata = {
        "schema": "PaperEWCExactDiagonalFisher-v1",
        "paper": "Kirkpatrick et al., PNAS 2017, arXiv:1612.00796",
        "formula": "mean_n square(grad_theta L_A(x_n; theta_A_star))",
        "per_example": True,
        "batch_size": 1,
        "steps": steps,
        "seed": seed,
        "task_a_repo_id": config.data.repo_id,
        "theta_a_star": str(checkpoint),
        "theta_a_star_metadata_sha256": hashlib.sha256(
            (checkpoint / "_METADATA").read_bytes()
        ).hexdigest(),
        "normalization_asset_id": config.data.assets.asset_id,
        "full_parameter": True,
        "lora": False,
        "factorized_or_low_rank_fisher": False,
        "tensor_count": len(values),
        "parameter_count": total_params,
        "nonzero_count": int(nonzero),
        "fisher_mass": float(total_mass),
        "mean_fisher": float(total_mass) / total_params,
        "host": platform.node(),
        "jax_devices": [str(device) for device in jax.devices()],
        "orbax_path": str(out),
    }
    manifest = out.parent / "manifest.json"
    with manifest.open("x") as stream:
        stream.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    logging.info("saved exact Fisher to %s", out)
    logging.info("manifest: %s", manifest)
