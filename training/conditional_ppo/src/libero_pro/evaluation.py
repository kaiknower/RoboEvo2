"""Complete, explicit per-task evaluation and validation-only checkpoint selection."""
import json
from pathlib import Path
from .campaign import CAPS, SPLITS, Condition, conditions
from .protocol import bddl_language
from .runtime import ROOT, setup_runtime


def summarize(records, condition, split, model_path):
    Condition.parse(condition)
    expected = {(t, i) for t in range(10) for i in SPLITS[split]}
    keys = [(r["task_id"], r["init_id"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate evaluation episodes; no headline success rate")
    if any(type(r["success"]) is not bool for r in records):
        raise ValueError("Success must come from the simulator as a boolean")
    per_task = []
    for task_id in range(10):
        rows = [r for r in records if r["task_id"] == task_id]
        successes = sum(r["success"] for r in rows)
        per_task.append(dict(task_id=task_id, successes=successes, episodes=len(rows),
                             success_rate=successes / len(rows)))
    return dict(schema_version=1, condition=condition, split=split, model_path=str(Path(model_path).resolve()),
                episodes=len(records), successes=sum(r["success"] for r in records),
                success_rate=sum(t["success_rate"] for t in per_task) / 10,
                reward_coverage=sum(t["successes"] > 0 for t in per_task) / 10,
                per_task=per_task, protocol="libero_pro_ppo_v1", complete=True)


def select_best(paths, condition):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    for r in reports:
        if r.get("split") != "validation" or r.get("condition") != condition:
            raise ValueError("Selection requires validation reports for one identical condition")
        if not r.get("complete") or r.get("episodes") != 100 or r.get("protocol") != "libero_pro_ppo_v1":
            raise ValueError("Incomplete or mismatched validation protocol")
    if not reports:
        raise ValueError("No validation candidates")
    return max(reports, key=lambda r: r["success_rate"])


def evaluate(condition, model_path, baseline, split, output, device="cuda", seed=42, policy=None,
             smoke=False):
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test only")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    setup_runtime(output.parent / (output.stem + "-runtime"))
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.libero.utils import quat2axisangle
    if policy is None:
        from rlinf.models.embodiment.openpi import get_model
        from .training import build_config
        cfg = build_config(condition, baseline)
        cfg_path = output.parent / (output.stem + "-runtime") / "evaluate.yaml"
        OmegaConf.save(OmegaConf.create(cfg), cfg_path)
        with initialize_config_dir(config_dir=str(cfg_path.parent), version_base="1.1"):
            resolved = compose(config_name="evaluate")
        resolved.actor.model.model_path = str(Path(model_path).resolve())
        model = get_model(resolved.actor.model).to(device).eval()
    else:
        model = policy
    c = Condition.parse(condition)
    suite = benchmark.get_benchmark(c.benchmark)()
    if suite.get_num_tasks() != 10:
        raise ValueError("Expected all ten tasks; no silent exclusions")
    records = []
    task_ids = range(1) if smoke else range(10)
    init_ids = SPLITS[split][:1] if smoke else SPLITS[split]
    action_cap = 10 if smoke else CAPS[c.suite]
    with output.with_suffix(".jsonl").open("x") as stream:
        for tid in task_ids:
            task = suite.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            language = bddl_language(bddl)
            init_states = suite.get_task_init_states(tid)
            if max(SPLITS[split]) >= len(init_states):
                raise ValueError(f"Missing initial states for task {tid}")
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                for init_id in init_ids:
                    episode_seed = seed + tid * 1000 + init_id
                    torch.manual_seed(episode_seed)
                    np.random.seed(episode_seed)
                    env.seed(episode_seed)
                    env.reset()
                    obs = env.set_init_state(init_states[init_id])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    success = False
                    steps = 0
                    while steps < action_cap and not success:
                        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]])
                        batch = dict(main_images=torch.from_numpy(obs["agentview_image"][::-1, ::-1].copy())[None],
                            wrist_images=torch.from_numpy(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())[None],
                            states=torch.from_numpy(state.astype(np.float32))[None], task_descriptions=[language],
                            extra_view_images=None)
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                        with torch.inference_mode():
                            actions, _ = model.predict_action_batch(batch, mode="eval", compute_values=False)
                        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
                        for action in actions[0, :5, :7]:
                            obs, _, done, _ = env.step(action)
                            steps += 1
                            success = bool(done)
                            if success or steps >= action_cap:
                                break
                    row = dict(task_id=tid, init_id=init_id, seed=episode_seed, success=success, steps=steps, language=language)
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(f"{condition} {split} task={tid} init={init_id} success={success}", flush=True)
            finally:
                env.close()
    if smoke:
        report = dict(schema_version=1, condition=condition, split=split,
                      model_path=str(Path(model_path).resolve()), episodes=len(records),
                      protocol="libero_pro_ppo_smoke_v1", complete=False, smoke=True,
                      action_cap=action_cap, checks_passed=True)
    else:
        report = summarize(records, condition, split, model_path)
    report.update(seed=seed, sampling=dict(mode="ode", denoising_steps=3, executed_chunk=5))
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def aggregate(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    names = [r["condition"] for r in reports]
    if len(names) != 8 or set(names) != {c.name for c in conditions()}:
        raise ValueError("Exactly one report per suite × axis is required")
    if any(not r.get("complete") or r["episodes"] != 100 or r["split"] != "test" or r["protocol"] != "libero_pro_ppo_v1" for r in reports):
        raise ValueError("Require complete matching test reports")
    return dict(success_rate=sum(r["success_rate"] for r in reports) / 8,
                episodes=sum(r["episodes"] for r in reports), conditions=reports)
