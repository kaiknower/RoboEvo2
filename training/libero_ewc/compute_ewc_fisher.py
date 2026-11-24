#!/usr/bin/env python3
"""Precompute the factorized diagonal Fisher for EWC on the v8 LoRA recipe.

EWC is applied to each LoRA site's effective weight update dW = a @ b
(scaling alpha / rank = 1.0), weighted by the diagonal Fisher F of the base weight:

    L_ewc = 0.5 * lambda * sum_ij F_ij * dW_ij^2

F is estimated at the task-A optimum (adapters zeroed) on task-A data
(physical-intelligence/libero). Each site stores the rank-1 factorization
F_ij ~= u_i v_j / mass with u_i = sum_j G_ij, v_j = sum_i G_ij,
mass = sum_ij G_ij and G = E[(dL/dW)^2], so the penalty is

    (1/mass) * sum_rs M_rs N_rs,  M_rs = sum_i u_i a_ir a_is,  N_rs = sum_j v_j b_rj b_sj

at O((I + J) * rank^2) per site. u is rescaled so that the mean Fisher weight
over all covered parameters is 1.

Run through run.py fisher with a LoRA EWC recipe.
"""

import logging
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

LORA_REGEX = ".*lora.*"


def base_path_for(lora_a_path: str) -> str:
    """Map '<stem>lora_a' to the base weight tensor it additively modifies.

    Two naming shapes exist (verified against the realized param tree):
      .../attn/q_einsum/lora_a          -> .../attn/q_einsum/w
      .../mlp/gating_einsum_lora_a      -> .../mlp/gating_einsum
    """
    stem = lora_a_path[: -len("lora_a")]
    return stem + "w" if stem.endswith("/") else stem.rstrip("_")


def compute(cfg: _config.TrainConfig, out_path: Path, steps: int) -> None:
    logging.info(f"Fisher data: {cfg.data.repo_id} (norm stats from v8 assets), batch={cfg.batch_size}")

    loader = _data_loader.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=steps)

    rng = jax.random.key(0)
    model = cfg.model.create(rng)
    graphdef, state = nnx.split(model)
    partial_params = cfg.weight_loader.load(state.to_pure_dict())
    state.replace_by_pure_dict(partial_params)
    state = nnx_utils.state_map(
        state, nnx_utils.PathRegex(LORA_REGEX), lambda p: p.replace(jnp.zeros_like(p.value))
    )
    n_lora = len(state.filter(nnx_utils.PathRegex(LORA_REGEX)).flat_state())
    if n_lora == 0:
        raise SystemExit("no LoRA leaves found -- wrong config?")
    logging.info(f"model at theta*: base weights loaded, {n_lora} LoRA leaves zeroed")

    flat_shapes = {k: v.shape for k, v in traverse_util.flatten_dict(state.to_pure_dict(), sep="/").items()}
    sites = {}
    for a_path in sorted(k for k in flat_shapes if k.endswith("lora_a")):
        w_path = base_path_for(a_path)
        if w_path not in flat_shapes:
            raise SystemExit(f"no base tensor for {a_path} (guessed {w_path})")
        sites[a_path] = w_path
    logging.info(f"{len(sites)} LoRA sites -> base tensors")
    for a, w in sites.items():
        logging.info(f"  {a}  {flat_shapes[a]} x {flat_shapes[a[:-1] + 'b']}  ->  {w} {flat_shapes[w]}")

    def loss_fn(m, rng, observation, actions):
        return jnp.mean(m.compute_loss(rng, observation, actions, train=False))

    @jax.jit
    def grad_sq(state, rng, observation, actions):
        m = nnx.merge(graphdef, state)
        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(m, rng, observation, actions)
        flat = traverse_util.flatten_dict(grads.to_pure_dict(), sep="/")
        return {w: jnp.square(flat[w].astype(jnp.float32)) for w in sites.values()}

    accum = None
    n = 0
    for i, (observation, actions) in enumerate(loader):
        step_rng = jax.random.fold_in(rng, i)
        g2 = grad_sq(state, step_rng, observation, actions)
        accum = g2 if accum is None else {k: accum[k] + g2[k] for k in accum}
        n += 1
        if (i + 1) % 25 == 0:
            logging.info(f"  fisher batch {i + 1}/{steps}")
        if n >= steps:
            break
    if accum is None:
        raise SystemExit("data loader produced no batches")
    accum = {k: v / n for k, v in accum.items()}
    logging.info(f"accumulated squared grads over {n} batches of {cfg.batch_size}")

    out = {}
    total_mass = 0.0
    total_size = 0
    for a_path, w_path in sites.items():
        g = np.asarray(accum[w_path], np.float64)
        u = g.sum(axis=-1)
        v = g.sum(axis=-2)
        mass = g.sum(axis=(-2, -1))
        out[f"{a_path}::u"] = u
        out[f"{a_path}::v"] = v
        out[f"{a_path}::mass"] = np.maximum(mass, 1e-30)
        total_mass += float(g.sum())
        total_size += int(g.size)
    z = total_mass / max(total_size, 1)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"expected a positive finite Fisher mean, got {z}")
    logging.info(f"global Fisher mean = {z:.6e} over {total_size} params; normalizing to 1.0")
    for k in list(out):
        if k.endswith("::u"):
            out[k] = (out[k] / z).astype(np.float32)
        else:
            out[k] = out[k].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as stream:
        np.savez(stream, **out)
    logging.info(f"wrote {out_path}: {len(sites)} sites, {len(out)} arrays")
