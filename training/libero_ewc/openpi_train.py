import dataclasses
import fcntl
import functools
import logging
import os
import platform
import tempfile
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_pref as _pi0_pref
import openpi.models.pi0_redirection as _pi0_redirection
import openpi.models.pi0_routed as _pi0_routed
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def paper_ewc_penalty(
    trainable: dict[str, at.Array],
    fisher: dict[str, at.Array],
    reference: dict[str, at.Array],
) -> tuple[at.Array, int]:
    """Exact dense EWC quadratic over a flat parameter tree.

    Kept as a standalone pure function so the no-factorization contract can be
    tested independently of the multi-billion-parameter training job.
    """

    if trainable.keys() != fisher.keys() or trainable.keys() != reference.keys():
        raise ValueError("full EWC Fisher/reference keys do not exactly match trainable parameters")
    penalty = jnp.zeros((), jnp.float32)
    for path, theta in trainable.items():
        importance = fisher[path]
        theta_star = reference[path]
        if theta.shape != importance.shape or theta.shape != theta_star.shape:
            raise ValueError(f"full EWC shape mismatch at {path}")
        delta = theta.astype(jnp.float32) - theta_star.astype(jnp.float32)
        penalty = penalty + jnp.sum(importance.astype(jnp.float32) * jnp.square(delta))
    return penalty, len(trainable)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    pref_batch: Any = None,
    cpd_anchor_params: Any = None,
    *,
    ewc_fisher: Any = None,
    ewc_reference: Any = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if config.cpd_enabled:
            if pref_batch is None:
                raise ValueError("cpd_enabled but no preference batch was threaded into train_step.")
            bc_rng, pref_rng = jax.random.split(rng)
            bc_loss = (
                jnp.mean(model.compute_loss(bc_rng, observation, actions, train=True))
                if config.cpd_lambda_bc != 0.0
                else jnp.zeros((), jnp.float32)
            )
            pref_obs = _model.Observation.from_dict(pref_batch)
            if config.cpd_lambda_pref != 0.0:
                pref_reference = _pi0_pref.ADAPTER_OFF
                if config.cpd_pref_use_anchor_reference:
                    if cpd_anchor_params is None:
                        raise ValueError(
                            "cpd_pref_use_anchor_reference requires loaded cpd_anchor_params"
                        )
                    pref_reference = _pi0_pref.make_explicit_lora_ref(model, cpd_anchor_params)
                out = _pi0_pref.compute_loss_pref(
                    model,
                    nnx.state(model),
                    pref_reference,
                    {
                        "observation": pref_obs,
                        "a_plus": pref_batch["a_plus"],
                        "a_minus": pref_batch["a_minus"],
                        "action_mask": pref_batch.get("action_mask"),
                    },
                    pref_rng,
                    beta=config.cpd_beta,
                    lambda_prox=config.cpd_lambda_prox,
                    margin=(pref_batch["margin"] if config.cpd_margin_enabled else None),
                    objective=config.cpd_objective,
                    margin_cap=config.cpd_margin_cap,
                    prox_mode=config.cpd_prox_mode,
                    prox_slack=config.cpd_prox_slack,
                    train=True,
                )
            else:
                zero = jnp.zeros((), jnp.float32)
                out = {
                    "total": zero,
                    "l_dpo": zero,
                    "l_prox": zero,
                    "pref_acc": zero,
                    "delta_mean": zero,
                    "r_plus_mean": zero,
                    "r_minus_mean": zero,
                }
            failure_bc = jnp.zeros((), jnp.float32)
            if config.cpd_lambda_failure_bc != 0.0:
                failure_rng = jax.random.fold_in(pref_rng, 1)
                failure_bc = jnp.mean(model.compute_loss(failure_rng, pref_obs, pref_batch["a_minus"], train=True))
            redirection = {
                key: jnp.zeros((), jnp.float32)
                for key in (
                    "total",
                    "attraction",
                    "suppression",
                    "correction",
                    "negative_distance",
                    "correction_distance",
                    "suppression_active",
                    "suppression_margin",
                    "correctable_fraction",
                )
            }
            predicted_clean = None
            if config.cpd_redirection_enabled or config.lambda_direction != 0.0:
                if not hasattr(model, "predict_clean_action_endpoint"):
                    raise ValueError("CPD endpoint losses require a flow model with endpoint prediction support.")
                redirection_rng = jax.random.fold_in(pref_rng, 2)
                if config.lambda_direction != 0.0:
                    _, endpoint_rng = jax.random.split(redirection_rng)
                    predicted_clean, _ = model.predict_clean_action_endpoint(
                        endpoint_rng, pref_obs, pref_batch["a_minus"], train=True
                    )
            if config.cpd_redirection_enabled:
                redirection = _pi0_redirection.compute_redirection_loss(
                    model,
                    redirection_rng,
                    pref_obs,
                    pref_batch["a_plus"],
                    pref_batch["a_minus"],
                    correctable_mask=pref_batch.get("correctable"),
                    action_mask=(pref_batch.get("action_mask") if config.cpd_redirection_use_action_mask else None),
                    lambda_attraction=config.cpd_redirection_lambda_attraction,
                    lambda_suppression=config.cpd_redirection_lambda_suppression,
                    lambda_correction=config.cpd_redirection_lambda_correction,
                    suppression_margin=config.cpd_redirection_margin,
                    suppression_action_dim_weights=config.cpd_redirection_suppression_action_dim_weights,
                    pair_separation_margin_scale=config.cpd_redirection_pair_margin_scale,
                    predicted_clean=predicted_clean,
                    train=True,
                )
            directional = None
            if config.lambda_direction != 0.0:
                directional_fn = (
                    _pi0_redirection.group_balanced_directional_margin_loss
                    if config.direction_group_balance
                    else _pi0_redirection.directional_margin_loss
                )
                directional_kwargs = {"rho": config.rho}
                if config.direction_group_balance:
                    directional_kwargs["gripper_weight"] = config.direction_gripper_weight
                directional = directional_fn(
                    predicted_clean,
                    pref_batch["a_plus"],
                    pref_batch["a_minus"],
                    action_mask=pref_batch.get("action_mask"),
                    **directional_kwargs,
                )
            anchor_loss = jnp.zeros((), jnp.float32)
            if config.cpd_anchor_lambda != 0.0:
                if cpd_anchor_params is None:
                    raise ValueError("cpd_anchor_lambda is nonzero but no explicit frozen reference was loaded")
                anchor_rng = jax.random.fold_in(pref_rng, 3)
                anchor_out = _pi0_routed.compute_loss_routed(
                    model,
                    None,
                    (observation, actions, None),
                    anchor_rng,
                    lambda_bc=0.0,
                    lambda_a=1.0,
                    base_params_or_flag=cpd_anchor_params,
                    global_anchor=True,
                    train=True,
                )
                anchor_loss = anchor_out["loss_anchor"]
            loss = (
                config.cpd_lambda_bc * bc_loss.astype(jnp.float32)
                + config.cpd_lambda_pref * out["total"].astype(jnp.float32)
                + config.cpd_lambda_failure_bc * failure_bc.astype(jnp.float32)
                + config.cpd_redirection_scale * redirection["total"].astype(jnp.float32)
                + config.cpd_anchor_lambda * anchor_loss.astype(jnp.float32)
            )
            if directional is not None:
                loss = _pi0_redirection.add_directional_term(
                    loss,
                    directional["loss"].astype(jnp.float32),
                    lambda_direction=config.cpd_redirection_scale * config.lambda_direction,
                )
            aux = {
                "loss_bc": bc_loss.astype(jnp.float32),
                "loss_pref": out["total"].astype(jnp.float32),
                "loss_failure_bc": failure_bc.astype(jnp.float32),
                "cpd_l_dpo": out["l_dpo"].astype(jnp.float32),
                "cpd_l_prox": out["l_prox"].astype(jnp.float32),
                "cpd_pref_acc": out["pref_acc"].astype(jnp.float32),
                "cpd_delta_mean": out["delta_mean"].astype(jnp.float32),
                "cpd_r_plus_mean": out["r_plus_mean"].astype(jnp.float32),
                "cpd_r_minus_mean": out["r_minus_mean"].astype(jnp.float32),
                "redir_total": redirection["total"].astype(jnp.float32),
                "redir_attraction": redirection["attraction"].astype(jnp.float32),
                "redir_suppression": redirection["suppression"].astype(jnp.float32),
                "redir_correction": redirection["correction"].astype(jnp.float32),
                "redir_negative_distance": redirection["negative_distance"].astype(jnp.float32),
                "redir_correction_distance": redirection["correction_distance"].astype(jnp.float32),
                "redir_suppression_active": redirection["suppression_active"].astype(jnp.float32),
                "redir_suppression_margin": redirection["suppression_margin"].astype(jnp.float32),
                "redir_correctable_fraction": redirection["correctable_fraction"].astype(jnp.float32),
                "cpd_anchor": anchor_loss.astype(jnp.float32),
            }
            if directional is not None:
                aux.update(
                    {
                        "loss_direction": directional["loss"].astype(jnp.float32),
                        "direction_q": directional["q"].astype(jnp.float32),
                        "direction_scale": directional["scale"].astype(jnp.float32),
                        "direction_active_fraction": directional["active_fraction"].astype(jnp.float32),
                    }
                )
                if config.direction_group_balance:
                    aux.update(
                        {
                            "direction_pose_q": directional["pose_q"].astype(jnp.float32),
                            "direction_pose_scale": directional["pose_scale"].astype(jnp.float32),
                            "direction_pose_active_fraction": directional["pose_active_fraction"].astype(jnp.float32),
                            "direction_gripper_q": directional["gripper_q"].astype(jnp.float32),
                            "direction_gripper_scale": directional["gripper_scale"].astype(jnp.float32),
                            "direction_gripper_active_fraction": directional["gripper_active_fraction"].astype(
                                jnp.float32
                            ),
                        }
                    )
            return loss.astype(jnp.float32), aux
        if config.lwf_enabled:
            out = _pi0_routed.compute_loss_routed(
                model,
                None,
                (observation, actions, None),
                rng,
                lambda_bc=1.0,
                lambda_a=config.lwf_lambda,
                global_anchor=True,
                train=True,
            )
            aux = {
                "loss_bc": out["loss_bc"],
                "loss_anchor": out["loss_anchor"],
            }
            return out["loss"], aux
        if config.ewc_enabled:
            if ewc_fisher is None:
                raise ValueError("ewc_enabled but no Fisher was threaded into train_step.")
            bc_loss = jnp.mean(model.compute_loss(rng, observation, actions, train=True))
            trainable = traverse_util.flatten_dict(
                nnx.state(model).filter(config.trainable_filter).to_pure_dict(), sep="/"
            )
            penalty = jnp.zeros((), jnp.float32)
            n_sites = 0
            if config.ewc_full_diagonal:
                if ewc_reference is None:
                    raise ValueError("ewc_full_diagonal requires the exact theta_A* reference tree")
                penalty, n_sites = paper_ewc_penalty(trainable, ewc_fisher, ewc_reference)
            else:
                for a_path, a in trainable.items():
                    if not a_path.endswith("lora_a") or f"{a_path}::u" not in ewc_fisher:
                        continue
                    b = trainable[a_path[:-1] + "b"]
                    u, v, mass = (ewc_fisher[f"{a_path}::{s}"] for s in ("u", "v", "mass"))
                    a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
                    gram_a = jnp.einsum("...ir,...is,...i->...rs", a32, a32, u)
                    gram_b = jnp.einsum("...rj,...sj,...j->...rs", b32, b32, v)
                    penalty = penalty + jnp.sum(jnp.sum(gram_a * gram_b, axis=(-2, -1)) / mass)
                    n_sites += 1
            if n_sites == 0:
                raise ValueError(
                    "ewc_enabled but the Fisher matched ZERO LoRA sites -- the keys in "
                    "ewc_fisher_path do not line up with the model state. Regenerate it "
                    "with compute_ewc_fisher.py for THIS config."
                )
            loss = bc_loss.astype(jnp.float32) + 0.5 * config.ewc_lambda * penalty
            aux = {
                "loss_bc": bc_loss.astype(jnp.float32),
                "ewc_penalty": penalty.astype(jnp.float32),
                "ewc_sites": jnp.asarray(n_sites, jnp.float32),
            }
            return loss.astype(jnp.float32), aux
        if config.vra_enabled:
            if observation.route is None:
                raise ValueError(
                    "vra_enabled but Observation.route is None -- the data pipeline is not "
                    "threading the route column (use LeRobotLiberoRouteDataConfig)."
                )
            out = _pi0_routed.compute_loss_routed(
                model,
                None,
                (observation, actions, observation.route),
                rng,
                lambda_bc=config.vra_lambda_bc,
                lambda_a=config.vra_lambda_a,
                global_anchor=config.vra_global,
                train=True,
            )
            aux = {k: out[k] for k in ("loss_bc", "loss_anchor", "frac_route0", "frac_route1", "frac_route2")}
            return out["loss"], aux
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        **aux,
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    pref_iter = None
    pref_batch = None
    if config.cpd_enabled:
        import openpi.training.pref_data as _pref_data

        pref_loader = _pref_data.create_pref_loader(config, sharding=data_sharding, shuffle=True)
        pref_iter = iter(pref_loader)
        pref_batch = next(pref_iter)
        logging.info(f"Initialized CPD preference loader:\n{training_utils.array_tree_to_info(pref_batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if (
        config.lwf_enabled
        or (config.ewc_enabled and not config.ewc_full_diagonal)
        or config.cpd_anchor_lambda != 0.0
        or config.cpd_pref_use_anchor_reference
    ):
        trainable_paths = list(
            traverse_util.flatten_dict(train_state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
        )
        non_lora = [p for p in trainable_paths if "lora" not in p]
        if non_lora:
            raise ValueError(
                f"EWC/CPD anchor requires freeze_filter=_G2E_FREEZE (LoRA leaves only), but "
                f"{len(non_lora)} non-LoRA tensors are trainable, e.g. {non_lora[:5]}. "
                "The adapter-off reference would drift with the student."
            )
        logging.info(f"EWC/CPD anchor freeze check OK: {len(trainable_paths)} trainable tensors, all LoRA")

    if config.ewc_full_diagonal:
        if not config.ewc_enabled:
            raise ValueError("ewc_full_diagonal=True requires ewc_enabled=True")
        all_paths = set(
            traverse_util.flatten_dict(train_state.params.to_pure_dict(), sep="/")
        )
        trainable_paths = set(
            traverse_util.flatten_dict(
                train_state.params.filter(config.trainable_filter).to_pure_dict(), sep="/"
            )
        )
        if trainable_paths != all_paths:
            raise ValueError(
                "dense full EWC requires every parameter to be trainable; "
                f"got {len(trainable_paths)}/{len(all_paths)} tensors"
            )
        lora_paths = sorted(path for path in all_paths if "lora" in path)
        if lora_paths:
            raise ValueError(
                "dense full EWC uses the native dense parameterization, not LoRA; "
                f"found {len(lora_paths)} LoRA tensors"
            )

    cpd_anchor_params = None
    if config.cpd_anchor_lambda != 0.0 or config.cpd_pref_use_anchor_reference:
        if not config.cpd_anchor_reference_checkpoint:
            raise ValueError(
                "cpd_anchor_lambda/cpd_pref_use_anchor_reference requires "
                "cpd_anchor_reference_checkpoint"
            )
        if config.cpd_pref_use_anchor_reference:
            init_path = getattr(config.weight_loader, "params_path", None)
            if init_path is None or str(init_path) != str(config.cpd_anchor_reference_checkpoint):
                raise ValueError(
                    "exact-parent preference reference requires weight_loader.params_path "
                    "to equal cpd_anchor_reference_checkpoint"
                )
        restored = _model.restore_params(epath.Path(config.cpd_anchor_reference_checkpoint), restore_type=np.ndarray)
        current_flat = traverse_util.flatten_dict(train_state.params.to_pure_dict(), sep="/")
        restored_flat = traverse_util.flatten_dict(restored, sep="/")
        expected_lora = {key for key in current_flat if "lora" in key}
        missing_lora = sorted(expected_lora - set(restored_flat))
        if missing_lora:
            raise ValueError(f"CPD anchor checkpoint is missing {len(missing_lora)} LoRA leaves: {missing_lora[:5]}")
        anchor_flat = {}
        for key in sorted(expected_lora):
            value = restored_flat[key]
            expected = current_flat[key]
            if value.shape != expected.shape:
                raise ValueError(f"CPD anchor shape mismatch at {key}: {value.shape} != {expected.shape}")
            anchor_flat[key] = value.astype(expected.dtype) if value.dtype != expected.dtype else value
        del restored, restored_flat
        cpd_anchor_params = traverse_util.unflatten_dict(anchor_flat, sep="/")
        cpd_anchor_params = jax.device_put(cpd_anchor_params, replicated_sharding)
        logging.info(
            "CPD anchor: loaded %d frozen initialization LoRA tensors from %s",
            len(anchor_flat),
            config.cpd_anchor_reference_checkpoint,
        )

    ewc_fisher = None
    ewc_reference = None
    ewc_fisher_sharding = None
    ewc_reference_sharding = None
    if config.ewc_enabled:
        if not config.ewc_fisher_path:
            raise ValueError("ewc_enabled but ewc_fisher_path is unset.")
        if config.ewc_full_diagonal:
            fisher_tree = _model.restore_params(config.ewc_fisher_path, restore_type=np.ndarray)
            live_trainable_tree = train_state.params.filter(config.trainable_filter).to_pure_dict()
            fisher_flat = traverse_util.flatten_dict(fisher_tree, sep="/")
            reference_shape = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), live_trainable_tree
            )
            reference_tree = _load_weights_and_validate(config.weight_loader, reference_shape)
            reference_flat = traverse_util.flatten_dict(reference_tree, sep="/")
            if fisher_flat.keys() != reference_flat.keys():
                missing = sorted(reference_flat.keys() - fisher_flat.keys())
                extra = sorted(fisher_flat.keys() - reference_flat.keys())
                raise ValueError(
                    "dense EWC Fisher tree mismatch: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            for path, value in reference_flat.items():
                if fisher_flat[path].shape != value.shape:
                    raise ValueError(
                        f"dense EWC Fisher shape mismatch at {path}: "
                        f"{fisher_flat[path].shape} != {value.shape}"
                    )

            fisher_shape = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32), fisher_tree
            )
            fisher_tree_sharding = sharding.fsdp_sharding(fisher_shape, mesh)
            reference_tree_sharding = sharding.fsdp_sharding(reference_shape, mesh)
            fisher_tree = jax.device_put(
                jax.tree.map(lambda x: np.asarray(x, np.float32), fisher_tree),
                fisher_tree_sharding,
            )
            reference_tree = jax.device_put(reference_tree, reference_tree_sharding)
            jax.block_until_ready((fisher_tree, reference_tree))
            ewc_fisher = traverse_util.flatten_dict(fisher_tree, sep="/")
            ewc_reference = traverse_util.flatten_dict(reference_tree, sep="/")
            ewc_fisher_sharding = jax.tree.map(lambda x: x.sharding, ewc_fisher)
            ewc_reference_sharding = jax.tree.map(lambda x: x.sharding, ewc_reference)
            total = sum(int(v.size) for v in ewc_fisher.values())
            logging.info(
                "Dense EWC: loaded exact full diagonal Fisher and theta_A* for "
                f"{len(ewc_fisher)} tensors / {total} parameters"
            )
        else:
            with np.load(config.ewc_fisher_path) as npz:
                ewc_fisher = {k: jnp.asarray(npz[k], jnp.float32) for k in npz.files}
            total = sum(int(v.size) for v in ewc_fisher.values())
            logging.info(f"EWC: loaded diagonal Fisher for {len(ewc_fisher)} tensors / {total} params")

    if config.cpd_enabled and cpd_anchor_params is not None:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    elif config.cpd_enabled:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    elif config.ewc_full_diagonal:
        def paper_ewc_step(rng, state, batch, fisher, reference):
            return train_step(
                config,
                rng,
                state,
                batch,
                ewc_fisher=fisher,
                ewc_reference=reference,
            )

        ptrain_step = jax.jit(
            paper_ewc_step,
            in_shardings=(
                replicated_sharding,
                train_state_sharding,
                data_sharding,
                ewc_fisher_sharding,
                ewc_reference_sharding,
            ),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    elif config.ewc_enabled:
        ptrain_step = jax.jit(
            functools.partial(train_step, config, ewc_fisher=ewc_fisher),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if config.cpd_enabled and cpd_anchor_params is not None:
                train_state, info = ptrain_step(train_rng, train_state, batch, pref_batch, cpd_anchor_params)
            elif config.cpd_enabled:
                train_state, info = ptrain_step(train_rng, train_state, batch, pref_batch)
            elif config.ewc_full_diagonal:
                train_state, info = ptrain_step(
                    train_rng, train_state, batch, ewc_fisher, ewc_reference
                )
            else:
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)
        if config.cpd_enabled:
            pref_batch = next(pref_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            if config.checkpoint_params_only:
                with open(os.path.join(tempfile.gettempdir(), "openpi_params_only_checkpoint.lock"), "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    _checkpoints.save_state(
                        checkpoint_manager,
                        train_state,
                        data_loader,
                        step,
                        params_only=True,
                    )
                    checkpoint_manager.wait_until_finished()
            else:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
