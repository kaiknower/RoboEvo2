"""Compose genuine RLinf PPO runs; each condition has its own optimizer/checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from .campaign import CAPS, SPLITS, Condition
from .runtime import EMBODIED_PATH, setup_runtime


def build_config(condition, model_path, output_root="runs", iterations=200, smoke=False,
                 micro_batch_size=None, smoke_world_size=1):
    c = Condition.parse(condition)
    model_path = Path(model_path).resolve()
    config_dir = EMBODIED_PATH / "config"
    base = config_dir / "libero_spatial_ppo_openpi_pi05.yaml"
    cfg = yaml.safe_load(base.read_text())
    cfg["defaults"][0:2] = ["env/libero_spatial@env.train", "env/libero_spatial@env.eval"]
    cfg["defaults"].append("_self_")
    cfg["hydra"]["searchpath"] = [f"file://{config_dir}"]
    cfg["runner"].update(max_epochs=iterations, save_interval=1, val_check_interval=-1,
        libero_pro_validation=dict(condition=c.name, baseline=str(model_path), smoke=smoke))
    cfg["runner"]["logger"].update(log_path=str(Path(output_root).resolve()), experiment_name=c.name,
                                     project_name="libero-pro", logger_backends=["tensorboard"])
    cfg["algorithm"]["bootstrap_type"] = "none"
    for split in ("train", "eval"):
        e = cfg["env"][split]
        e.update(task_suite_name=c.benchmark, libero_pro=True,
                 trial_ids=list(SPLITS["train" if split == "train" else "validation"]),
                 task_id_filter=list(range(10)), specific_reset_id=None, settle_steps=10,
                 auto_reset=True, ignore_terminations=False,
                 max_episode_steps=CAPS[c.suite], max_steps_per_rollout_epoch=1200,
                 use_rel_reward=False, use_step_penalty=False, seed=42 if split == "train" else 43,
                 video_cfg=dict(save_video=False, info_on_video=False, video_base_dir="${runner.logger.log_path}/video"))
    cfg["env"]["eval"].update(total_num_envs=16, rollout_epoch=1)
    cfg["actor"]["model"].update(model_path=str(model_path),
        openpi_data=dict(norm_stats_path=str(model_path / "assets/roboevo2/libero_expert_v8/norm_stats.json")))
    cfg["actor"]["model"]["openpi"].update(noise_method="flow_sde", noise_level=0.5,
        joint_logprob=True, full_path_likelihood=True, train_expert_only=True)
    cfg["rollout"]["model"]["model_path"] = str(model_path)
    cfg["actor"]["fsdp_config"]["sharding_strategy"] = "full_shard"
    if smoke:
        cfg["env"]["train"].update(total_num_envs=2 * smoke_world_size, rollout_epoch=1,
                                    max_steps_per_rollout_epoch=20)
        cfg["env"]["eval"].update(total_num_envs=2 * smoke_world_size)
        cfg["actor"].update(micro_batch_size=1, global_batch_size=8 * smoke_world_size)
    if micro_batch_size is not None:
        if micro_batch_size < 1:
            raise ValueError("Micro batch size must be positive")
        cfg["actor"]["micro_batch_size"] = micro_batch_size
    return cfg


def validate_config(cfg, actor_world_size=None):
    train, validation = cfg.env.train, cfg.env.eval
    if set(train.trial_ids) & set(validation.trial_ids):
        raise ValueError("Train/validation init states overlap")
    if set(train.trial_ids) & set(SPLITS["test"]):
        raise ValueError("Training would consume held-out test states")
    if cfg.actor.model.num_action_chunks != 5 or cfg.actor.model.num_steps != 3:
        raise ValueError("This campaign uses five executed actions and three denoising steps")
    if train.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks:
        raise ValueError("Rollout horizon must be divisible by chunk length")
    if not cfg.actor.model.openpi.joint_logprob:
        raise ValueError("PPO needs the full denoising path likelihood")
    if actor_world_size is not None:
        divisor = cfg.actor.micro_batch_size * actor_world_size
        if cfg.actor.global_batch_size % divisor:
            raise ValueError("Global batch must be divisible by micro batch × actor GPU count")
        samples = train.total_num_envs * train.rollout_epoch * (
            train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks)
        if samples % cfg.actor.global_batch_size:
            raise ValueError("Rollout sample count must be divisible by global batch size")


def train(condition, model_path, output_root, iterations, smoke=False, dry_run=False, resume=None,
          micro_batch_size=None):
    run_dir = Path(output_root).resolve() / condition
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    if smoke and resume is not None:
        raise ValueError("Smoke runs require a fresh output directory")
    if not dry_run and (run_dir / "launch.json").exists() and resume is None:
        raise FileExistsError(f"Existing run {run_dir}; use --resume or a new --output-root")
    if resume is not None and not dry_run:
        checkpoints = sorted((run_dir / "checkpoints").glob("global_step_*"),
                             key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        if not checkpoints or Path(resume).resolve() != checkpoints[-1].resolve():
            raise ValueError("Resume the latest checkpoint in this condition's existing run")
        if not (run_dir / "evaluation/validation_0000.json").is_file():
            raise ValueError("Resume requires this run's original baseline validation report")
    setup_runtime(run_dir)
    world_size = None
    try:
        import torch
        world_size = torch.cuda.device_count() or None
    except ImportError:
        if not dry_run:
            raise
    if not dry_run and world_size is None:
        raise RuntimeError("Training requires CUDA PyTorch and visible NVIDIA GPUs")
    cfg = build_config(condition, model_path, output_root, iterations, smoke,
                       micro_batch_size, smoke_world_size=world_size or 1)
    cfg["runner"]["resume_dir"] = str(Path(resume).resolve()) if resume else None
    config_path = run_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(run_dir), version_base="1.1"):
        composed = compose(config_name="train")
    validate_config(composed, actor_world_size=world_size)
    if dry_run:
        print(config_path)
        return
    model_path = Path(model_path).resolve()
    if not (model_path / "model.safetensors").is_file() or not (model_path / "conversion.json").is_file():
        raise FileNotFoundError("Run convert first: model.safetensors and conversion.json required")
    if os.environ.get("LIBERO_INIT_ID_ALLOWLIST"):
        raise ValueError("Unset LIBERO_INIT_ID_ALLOWLIST; splits are per environment")
    command = [sys.executable, str(EMBODIED_PATH / "train_embodied_agent.py"),
               "--config-path", str(run_dir), "--config-name", "train"]
    (run_dir / "launch.json").write_text(json.dumps(dict(condition=condition, model=str(model_path),
        command=command, smoke=smoke, resume=resume), indent=2))
    subprocess.run(command, check=True, env=os.environ.copy())
