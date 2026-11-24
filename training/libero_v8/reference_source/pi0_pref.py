"""CPD: Counterfactual Preference Distillation loss for pi0/pi05 flow-matching models.

Each preference sample holds the takeover-point observation ``s`` and two action
chunks: ``a_plus`` (verified correction, preferred) and ``a_minus`` (failed policy
chunk, dispreferred). Per-sample flow-matching losses are the standard Pi0
objective reduced over the action-horizon axis. All four evaluations (theta/ref x
a_plus/a_minus) share the same (t, eps) draw and image-preprocessing rng, which
reduces the variance of the loss differences.

    Delta_plus  = L_FM^theta(a+|s) - L_FM^ref(a+|s)          (per sample)
    Delta_minus = L_FM^theta(a-|s) - L_FM^ref(a-|s)
    Delta       = Delta_plus - Delta_minus
    Delta_used  = max(Delta + margin, -margin_cap)            (both optional)

    objective="ipo":  L_pref = (Delta_used + 1/(2 beta))^2    (paper: L_gap)
    objective="dpo":  L_pref = softplus(beta * Delta_used)
    prox_mode="onesided": L_prox = relu(Delta_plus)^2 + relu(Delta_minus - slack)^2
    prox_mode="sym":      L_prox = Delta_plus^2 + Delta_minus^2
    total = mean(L_pref) + lambda_prox * mean(L_prox)

Implicit rewards are r(a|s) = -beta * [L_FM^theta(a|s) - L_FM^ref(a|s)].

Reference model ("ref"):
  * ``ref_params_or_flag = "adapter_off"`` (or ``True``): the LoRA adapter-off
    forward. LoRA enters the forward additively, so zeroing every ``.*lora.*`` leaf
    reproduces the base model exactly without a second copy of the weights. The
    non-LLM projections (action_in_proj, action_out_proj, time_mlp_in/out) stay
    trainable under the default freeze filter; freeze them as well if the
    adapter-off model must equal the base model exactly.
  * ``ref_params_or_flag = <nnx.State>``: explicit frozen reference parameters.
  In both cases the reference parameters are wrapped in ``stop_gradient``.

Typical usage:

    graphdef, state = nnx.split(model)
    out = compute_loss_pref(graphdef, state, "adapter_off", pref_batch, rng,
                            beta=0.5, lambda_prox=0.5, objective="ipo",
                            margin_cap=1.0, prox_mode="onesided", prox_slack=0.3)
    out["total"]  # mixed with the BC loss: L = L_FM + lambda_pref * total

The module does not modify pi0.py and works with any BaseModel whose
``compute_loss(rng, obs, actions, train=...)`` returns a per-sample [*b, ah] loss.
"""

import logging

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

ADAPTER_OFF = "adapter_off"

_LORA_PATH_REGEX = ".*lora.*"


def make_adapter_off_ref(params: nnx.State) -> nnx.State:
    """Build reference params for the adapter-off (LoRA) forward: zero all LoRA leaves.

    Exactness argument: the LoRA contribution is strictly additive on top of the base
    weight path (lora.py Einsum.__call__ L59-63 and FeedForward._dot L144-148), so with
    lora_a = lora_b = 0 the forward is bit-for-bit the base model's forward.
    """
    lora_filter = nnx_utils.PathRegex(_LORA_PATH_REGEX)
    n_lora = len(params.filter(lora_filter).flat_state())
    if n_lora == 0:
        logger.warning(
            "make_adapter_off_ref: no LoRA parameters found in the state; "
            "the reference will be identical to theta (Delta will be 0)."
        )
    return nnx_utils.state_map(params, lora_filter, lambda p: p.replace(jnp.zeros_like(p.value)))


def make_explicit_lora_ref(model: nnx.Module, lora_params) -> nnx.State:
    """Build a frozen full reference state from a LoRA-only checkpoint.

    The non-LoRA arrays are shared with ``model`` and only the small LoRA leaves
    are replaced, so this does not materialize a second multi-billion-parameter
    backbone.  Missing, extra, or shape-mismatched leaves fail closed: silently
    falling back to adapter-off would change the scientific reference policy.
    """
    _, state = nnx.split(model)
    current_flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    if isinstance(lora_params, nnx.State):
        lora_params = lora_params.to_pure_dict()
    if not isinstance(lora_params, dict):
        raise TypeError(f"explicit LoRA reference must be a dict or nnx.State, got {type(lora_params)}")
    provided_flat = traverse_util.flatten_dict(lora_params, sep="/")
    expected_lora = {key for key in current_flat if "lora" in key}
    if not expected_lora:
        raise ValueError("explicit LoRA reference requested for a model with no LoRA leaves")
    provided = set(provided_flat)
    missing = sorted(expected_lora - provided)
    unexpected = sorted(provided - expected_lora)
    if missing or unexpected:
        raise ValueError(
            "explicit LoRA reference keys do not exactly match the model: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )
    for key in sorted(expected_lora):
        if provided_flat[key].shape != current_flat[key].shape:
            raise ValueError(
                f"explicit LoRA reference shape mismatch at {key}: "
                f"{provided_flat[key].shape} != {current_flat[key].shape}"
            )

    full_ref = {
        key: (provided_flat[key] if key in expected_lora else value)
        for key, value in current_flat.items()
    }
    full_ref = {key: jax.lax.stop_gradient(value) for key, value in full_ref.items()}
    state.replace_by_pure_dict(traverse_util.unflatten_dict(full_ref, sep="/"))
    return state


def fm_per_sample_loss(model, rng, observation, actions, *, action_mask=None, train: bool = False):
    """Per-sample flow-matching loss L_FM(a|s), shape [*b].

    Reduces Pi0.compute_loss, which returns the per-sample per-horizon-step
    loss [*b, ah]. Reusing the same rng gives a shared (t, eps): the
    (preprocess, noise, time) keys are derived deterministically from `rng` and the
    noise shape depends only on `actions.shape`, identical for a_plus/a_minus.
    """
    chunked = model.compute_loss(rng, observation, actions, train=train)
    if action_mask is None:
        return jnp.mean(chunked, axis=-1)
    mask = jnp.asarray(action_mask, dtype=chunked.dtype)
    if mask.shape[-1:] != chunked.shape[-1:]:
        raise ValueError(
            f"action_mask horizon must match FM loss horizon: mask={mask.shape}, loss={chunked.shape}"
        )
    try:
        mask = jnp.broadcast_to(mask, chunked.shape)
    except ValueError as exc:
        raise ValueError(
            f"action_mask must broadcast to per-step FM loss: mask={mask.shape}, loss={chunked.shape}"
        ) from exc
    valid_steps = jnp.sum(mask, axis=-1)
    valid_steps = jnp.maximum(valid_steps, jnp.asarray(1, dtype=chunked.dtype))
    return jnp.sum(chunked * mask, axis=-1) / valid_steps


def _unpack_pref_batch(pref_batch):
    """Accept observation, plus/minus, optional margin, and optional action mask."""
    if isinstance(pref_batch, dict):
        return (
            pref_batch["observation"],
            pref_batch["a_plus"],
            pref_batch["a_minus"],
            pref_batch.get("margin"),
            pref_batch.get("action_mask"),
        )
    if isinstance(pref_batch, (tuple, list)):
        observation, a_plus, a_minus, *rest = pref_batch
        return (
            observation,
            a_plus,
            a_minus,
            (rest[0] if rest else None),
            (rest[1] if len(rest) > 1 else None),
        )
    return (
        pref_batch.observation,
        pref_batch.a_plus,
        pref_batch.a_minus,
        getattr(pref_batch, "margin", None),
        getattr(pref_batch, "action_mask", None),
    )


def _is_adapter_off_flag(ref_params_or_flag) -> bool:
    return (ref_params_or_flag is True) or (
        isinstance(ref_params_or_flag, str) and ref_params_or_flag == ADAPTER_OFF
    )


def compute_loss_pref(
    model,
    params: nnx.State,
    ref_params_or_flag,
    pref_batch,
    rng,
    *,
    beta: float = 0.5,
    lambda_prox: float = 0.1,
    margin=None,
    objective: str = "dpo",
    margin_cap: float | None = None,
    prox_mode: str = "sym",
    prox_slack: float = 0.0,
    train: bool = False,
) -> dict:
    """CPD preference loss on one preference batch.

    Args:
      model: a Pi0 `nnx.Module` (its graphdef is taken) or an `nnx.GraphDef` directly.
      params: FULL `nnx.State` for the policy theta (as produced by `nnx.split(model)`
        / `nnx.state(model)`); must match the graphdef.
      ref_params_or_flag: either `"adapter_off"` / `True` (LoRA route: reference =
        params with every `.*lora.*` leaf zeroed — see module docstring for the
        exactness argument and caveat), or an explicit full `nnx.State` of frozen
        reference parameters. Always evaluated under `stop_gradient`.
      pref_batch: dict/tuple/object with `observation` (openpi Observation), `a_plus`,
        `a_minus` ([*b, ah, ad] action chunks), optionally per-pair `margin` [*b],
        and optionally `action_mask` [*b, ah]. A missing mask means all horizon
        positions are valid.
      rng: PRNG key. Split exactly ONCE; the single resulting key is reused for all
        four FM evaluations => shared (t, eps) and shared image preprocessing.
      beta: preference temperature; the IPO target gap is 1/(2*beta).
      lambda_prox: coefficient of the proximal term.
      margin: optional per-pair scalar offset (ODPO-style, verifier-graded progress
        difference), broadcastable to [*b]. Overrides `pref_batch["margin"]` if both
        given. Positive margin makes the pair harder (demands Delta <= -margin).
      train: passed through to compute_loss (enables train-time image augmentation,
        shared across the four evaluations via the shared rng).

    Returns a dict of scalars (plus per-sample diagnostics):
      total, l_dpo, l_prox, r_plus_mean, r_minus_mean, delta_mean, delta_used_mean,
      pref_acc, l_fm_theta_plus_mean, l_fm_theta_minus_mean, l_fm_ref_plus_mean,
      l_fm_ref_minus_mean, and per-sample arrays r_plus, r_minus, delta.
    """
    graphdef = nnx.graphdef(model) if isinstance(model, nnx.Module) else model
    observation, a_plus, a_minus, batch_margin, action_mask = _unpack_pref_batch(pref_batch)
    if margin is None:
        margin = batch_margin

    if _is_adapter_off_flag(ref_params_or_flag):
        ref_params = make_adapter_off_ref(params)
    else:
        ref_params = ref_params_or_flag
    ref_params = jax.tree.map(jax.lax.stop_gradient, ref_params)

    model_theta = nnx.merge(graphdef, params)
    model_ref = nnx.merge(graphdef, ref_params)

    (shared_rng,) = jax.random.split(rng, 1)

    l_theta_plus = fm_per_sample_loss(
        model_theta, shared_rng, observation, a_plus, action_mask=action_mask, train=train
    )
    l_theta_minus = fm_per_sample_loss(
        model_theta, shared_rng, observation, a_minus, action_mask=action_mask, train=train
    )
    l_ref_plus = fm_per_sample_loss(
        model_ref, shared_rng, observation, a_plus, action_mask=action_mask, train=train
    )
    l_ref_minus = fm_per_sample_loss(
        model_ref, shared_rng, observation, a_minus, action_mask=action_mask, train=train
    )

    delta_plus = l_theta_plus - l_ref_plus
    delta_minus = l_theta_minus - l_ref_minus
    delta = delta_plus - delta_minus

    delta_used = delta
    if margin is not None:
        delta_used = delta + jnp.asarray(margin, dtype=delta.dtype)

    if margin_cap is not None:
        delta_used = jnp.maximum(delta_used, -jnp.asarray(margin_cap, dtype=delta.dtype))

    if objective == "dpo":
        l_pref = jnp.mean(jax.nn.softplus(beta * delta_used))
    elif objective == "ipo":
        target = 1.0 / (2.0 * beta)
        l_pref = jnp.mean(jnp.square(delta_used + target))
    else:
        raise ValueError(f"unknown objective {objective!r} (expected 'dpo' or 'ipo')")

    if prox_mode == "sym":
        l_prox = jnp.mean(jnp.square(delta_plus) + jnp.square(delta_minus))
    elif prox_mode == "onesided":
        over = jax.nn.relu(delta_minus - prox_slack)
        l_prox = jnp.mean(jnp.square(jax.nn.relu(delta_plus)) + jnp.square(over))
    else:
        raise ValueError(f"unknown prox_mode {prox_mode!r} (expected 'sym' or 'onesided')")

    l_dpo = l_pref
    total = l_pref + lambda_prox * l_prox

    r_plus = -beta * delta_plus
    r_minus = -beta * delta_minus

    return {
        "total": total,
        "l_dpo": l_dpo,
        "l_prox": l_prox,
        "r_plus_mean": jnp.mean(r_plus),
        "r_minus_mean": jnp.mean(r_minus),
        "delta_mean": jnp.mean(delta),
        "delta_used_mean": jnp.mean(delta_used),
        "pref_acc": jnp.mean((delta < 0).astype(jnp.float32)),
        "l_fm_theta_plus_mean": jnp.mean(l_theta_plus),
        "l_fm_theta_minus_mean": jnp.mean(l_theta_minus),
        "l_fm_ref_plus_mean": jnp.mean(l_ref_plus),
        "l_fm_ref_minus_mean": jnp.mean(l_ref_minus),
        "r_plus": r_plus,
        "r_minus": r_minus,
        "delta": delta,
    }
