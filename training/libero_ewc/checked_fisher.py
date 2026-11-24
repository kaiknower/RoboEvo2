#!/usr/bin/env python3
"""Run the effective-weight Fisher estimator with per-batch validation."""

import logging
import os
import re


def flow_loss_with_draws(model, obs, actions, noise, time, *, inspect=False):
    """Evaluate Pi0.5 flow matching with explicit, prevalidated random draws."""
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import arfm_normalized_weights, make_attn_mask

    obs = model_lib.preprocess_observation(None, obs, train=False)
    expanded = time[..., None, None]
    noisy_actions = expanded * noise + (1 - expanded) * actions
    prefix, prefix_mask, prefix_roles = model.embed_prefix(obs)
    suffix, suffix_mask, suffix_roles, condition = model.embed_suffix(obs, noisy_actions, time)
    mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
    attention = make_attn_mask(mask, jnp.concatenate((prefix_roles, suffix_roles), axis=0))
    positions = jnp.cumsum(mask, axis=1) - 1
    (vision_text, action_features), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, condition]
    )
    velocity = model.action_out_proj(action_features[:, -model.action_horizon:])
    losses = jnp.mean(jnp.square(velocity - (noise - actions)), axis=-1)
    if obs.advantage is not None:
        import jax

        weights = arfm_normalized_weights(obs.advantage, model.arfm_alpha)
        losses *= jax.lax.stop_gradient(weights).astype(losses.dtype)[..., None]
    if inspect:
        return losses, {
            "noise": noise, "noisy_actions": noisy_actions, "prefix": prefix,
            "suffix": suffix, "time_condition": condition, "vision_text": vision_text,
            "action_features": action_features, "velocity": velocity, "loss": losses,
        }
    return losses


def fixed_draws(seed, num_batches, action_shape):
    import jax
    import jax.numpy as jnp
    import numpy as np

    def draw(key, index):
        _, noise_key, time_key = jax.random.split(jax.random.fold_in(key, index), 3)
        noise = jax.random.normal(noise_key, action_shape)
        times = jax.random.beta(time_key, 1.5, 1, action_shape[:-2]) * 0.999 + 0.001
        return noise, times

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        noise, times = jax.device_get(jax.jit(jax.vmap(draw, in_axes=(None, 0)))(
            jax.random.key(seed), jnp.arange(num_batches)
        ))
    if not np.isfinite(noise).all() or not (np.isfinite(times) & (times > 0) & (times < 1)).all():
        raise FloatingPointError("Invalid precomputed Pi0.5 flow-matching draws")
    return noise, times


def inspect_forward(graph, state, keys, obs, actions):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    def per_example(value):
        shape = (actions.shape[0], -1)
        value = value.astype(jnp.float32).reshape(shape)
        return dict(nonfinite=(~jnp.isfinite(value)).sum(axis=-1),
                    peak=jnp.max(jnp.abs(value), axis=-1))

    def trace(state, key, obs, actions):
        model = nnx.merge(graph, state)
        _, noise_key, time_key = jax.random.split(key, 3)
        noise = jax.random.normal(noise_key, actions.shape)
        time = jax.random.beta(time_key, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        _, intermediates = flow_loss_with_draws(model, obs, actions, noise, time, inspect=True)
        return {
            **{name: per_example(value) for name, value in intermediates.items()},
            "time": time,
            "token_ids_min": obs.tokenized_prompt.min(axis=-1),
            "token_ids_max": obs.tokenized_prompt.max(axis=-1),
        }

    def exact_loss(state, key, obs, actions):
        model = nnx.merge(graph, state)
        return jnp.mean(model.compute_loss(key, obs, actions, train=False), axis=-1)

    exact = jax.device_get(jax.jit(exact_loss)(state, keys, obs, actions))
    logging.info("Exact forward loss by sample: %s", exact)
    stages = jax.device_get(jax.jit(trace)(state, keys, obs, actions))
    for name, value in stages.items():
        logging.info("Forward stage %s: %s", name, value)


def compute(cfg, output, steps, probe_batch=0, *, shard_index=0, shard_count=1):
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.shared import nnx_utils
    from openpi.training import data_loader

    from compute_ewc_fisher import LORA_REGEX, base_path_for
    from merge_fisher_shards import publish

    loader = data_loader.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=steps)
    rng = jax.random.key(0)
    graph, state = nnx.split(cfg.model.create(rng))
    expected = flatten_dict(state.to_pure_dict(), sep="/")
    parameters = cfg.weight_loader.load(state.to_pure_dict())
    restored = flatten_dict(parameters, sep="/")
    if set(expected) != set(restored):
        raise ValueError(f"Fisher checkpoint parameter mismatch: {sorted(set(expected) ^ set(restored))}")
    state.replace_by_pure_dict(parameters)
    del parameters, restored
    state = nnx_utils.state_map(
        state, nnx_utils.PathRegex(LORA_REGEX), lambda p: p.replace(jnp.zeros_like(p.value))
    )
    sites = {name: base_path_for(name) for name in sorted(expected) if name.endswith("lora_a")}
    if len(sites) != 10:
        raise ValueError(f"Expected ten Fisher sites, found {len(sites)}")
    total_size = sum(expected[name].size for name in sites.values())
    del expected
    logging.info("Fisher anchor loaded. Estimating %d batches of %d on %d sites.",
                 steps, cfg.batch_size, len(sites))

    selected_weights = nnx.All(
        nnx.Param, nnx_utils.PathRegex("|".join(map(re.escape, sites.values())))
    )
    if len(state.filter(selected_weights).flat_state()) != len(sites):
        raise ValueError("Fisher gradient filter did not select exactly the effective weight sites")

    def loss_fn(model, obs, actions, noise, time):
        losses = flow_loss_with_draws(model, obs, actions, noise, time)
        return jnp.mean(losses), jnp.mean(losses, axis=-1)

    @jax.jit
    def measure(state, obs, actions, noise, time):
        model = nnx.merge(graph, state)
        gradients = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, selected_weights), has_aux=True
        )
        (loss, per_example), grads = gradients(model, obs, actions, noise, time)
        flat = flatten_dict(grads.to_pure_dict(), sep="/")
        marginals = {}
        for name in sites.values():
            gradient = flat[name].astype(jnp.float32)
            squares = jnp.square(gradient)
            marginals[name] = (
                squares.sum(axis=-1), squares.sum(axis=-2), jnp.max(jnp.abs(gradient))
            )
        return loss, per_example, marginals

    assigned = range(shard_index, steps, shard_count)
    logging.info("Fisher shard %d/%d assigned %d complete batches",
                 shard_index + 1, shard_count, len(assigned))
    totals = {}
    count = 0
    draws = None
    for index, (obs, actions) in enumerate(loader):
        if probe_batch:
            if index + 1 == probe_batch:
                logging.info("Probing deterministic Fisher batch %d", probe_batch)
                inspect_forward(graph, state, jax.random.fold_in(rng, index), obs, actions)
                return
            continue
        if index not in assigned:
            continue
        if draws is None:
            draws = fixed_draws(0, steps, actions.shape)
            logging.info("Validated all %d sets of flow-matching draws on CPU", steps)
        noise, times = (jax.device_put(values[index], actions.sharding) for values in draws)
        loss, per_example, marginals = jax.device_get(measure(state, obs, actions, noise, times))
        bad_sites = {name: float(peak) for name, (u, v, peak) in marginals.items()
                     if not (np.isfinite(peak) and np.isfinite(u).all() and np.isfinite(v).all())}
        if not np.isfinite(loss) or bad_sites:
            bad_inputs = [jax.tree_util.keystr(path) for path, value in
                          jax.tree_util.tree_flatten_with_path((obs, actions))[0]
                          if not bool(jnp.isfinite(value).all())]
            raise FloatingPointError(
                f"Fisher batch {index + 1}/{steps}: loss={loss}, per_example={per_example}, "
                f"nonfinite_inputs={bad_inputs}, nonfinite_gradient_sites_and_peaks={bad_sites}"
            )
        for name, (u, v, _) in marginals.items():
            if name not in totals:
                totals[name] = (u.astype(np.float64), v.astype(np.float64))
            else:
                row_total, column_total = totals[name]
                row_total += u
                column_total += v
        count += 1
        if count in (1, 2, 4, 8) or count % 25 == 0 or count == len(assigned):
            logging.info("Fisher batch %d/%d finite (%d/%d on shard, flow-matching loss %.6g)",
                         index + 1, steps, count, len(assigned), loss)
        if count == len(assigned):
            break
    if count != len(assigned):
        raise RuntimeError(f"Expected {len(assigned)} Fisher batches, processed {count}")
    if shard_count > 1:
        result = {
            "__batch_indices": np.asarray(assigned, dtype=np.int64),
            "__total_size": np.asarray(total_size, dtype=np.int64),
        }
        for site, name in sites.items():
            result[f"{site}::u_total"], result[f"{site}::v_total"] = totals[name]
        publish(output, result)
        logging.info("Fisher shard published %s (%d batches)", output, count)
        return
    mean = sum(u.sum() for u, _ in totals.values()) / (total_size * count)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"expected a positive finite Fisher mean, got {mean}")

    result = {}
    for site, name in sites.items():
        u, v = totals[name]
        u, v = u / count, v / count
        mass = u.sum(axis=-1)
        result[f"{site}::u"] = (u / mean).astype(np.float32)
        result[f"{site}::v"] = v.astype(np.float32)
        result[f"{site}::mass"] = np.maximum(mass, 1e-30).astype(np.float32)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Fisher values overflowed the on-disk format")
    publish(output, result)
    logging.info("Fisher published %s, %d sites, unnormalized mean %.8g", output, len(sites), mean)


def main():
    from run import load_config, parser

    cli = parser()
    args = cli.parse_args()
    if args.command != "fisher" or args.recipe != "pi05_libero_v8_ewc_lora":
        cli.error("Expected the rank-factorized LIBERO Fisher recipe")
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.check_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cfg = load_config(args)
    if args.check_only:
        print(f"{cfg.name}: checked Fisher inputs validated, dataset={cfg.data.repo_id}")
        return
    import jax
    from lerobot_datasets_compat import install

    if jax.default_backend() != "gpu":
        cli.error("Fisher estimation requires a CUDA JAX backend. Use --check-only on CPU.")
    install()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    probe_batch = int(os.environ.get("G2E_FISHER_PROBE_BATCH", "0"))
    steps = args.steps or 500
    if not 0 <= probe_batch <= steps:
        cli.error("Probe batch must be within the estimator's batch budget")
    shard_index, shard_count = map(int, os.environ.get("G2E_FISHER_SHARD", "0/1").split("/"))
    if not 0 <= shard_index < shard_count <= steps or probe_batch and shard_count > 1:
        cli.error("Invalid Fisher shard")
    compute(cfg, args.out, steps, probe_batch, shard_index=shard_index, shard_count=shard_count)


if __name__ == "__main__":
    main()
