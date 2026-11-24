"""OpenPI π0.5 PyTorch loss and checkpoint adapter for the LIBERO round worker.

Use --adapter evolution.libero_training:build with evolution.policy_update.
Input is a dense OpenPI PyTorch export with its v8 normalization assets.
D0 supplies synchronized robot demonstrations in the same trajectory manifest
format as evolution.libero_runtime, including a prompt, images and executed actions.
The default trainable subset is the action expert and its projection/time heads.
Set trainable_prefixes in adapter config to choose another subset.
"""
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .evidence import action_digest


@lru_cache(maxsize=256)
def _episode(filename):
    path = Path(filename)
    meta = json.loads(path.read_text())
    if (meta["schema"] != "roboevo.libero.trajectory.v1" or meta["image_orientation"] != "policy"
            or meta["action_reference"] != "executed" or not meta["chunks"] or not meta["steps"]):
        raise ValueError(f"Incompatible or empty LIBERO trajectory {path}")
    actions = np.load(path.parent / meta["actions"], mmap_mode="r", allow_pickle=False)
    if actions.shape != (meta["steps"], 7) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid executed LIBERO actions in {path}")
    ends, total = [], 0
    for name in meta["chunks"]:
        with np.load(path.parent / name, allow_pickle=False) as chunk:
            applied = chunk["action"]
            stop = total + len(applied)
            if not len(applied) or not np.array_equal(applied, actions[total:stop]):
                raise ValueError(f"Chunk actions differ from the executed targets in {path}")
            total = stop
        ends.append(total)
    if total != meta["steps"]:
        raise ValueError(f"Wrong number or shape of executed LIBERO actions in {path}")
    return meta, ends, actions


def _frame(row, index, action):
    filename = Path(row["trajectory"])
    meta, ends, _ = _episode(str(filename))
    chunk_id = bisect_right(ends, index)
    local_index = index - (ends[chunk_id-1] if chunk_id else 0)
    if meta["instruction"] != row["goal"]:
        raise ValueError("Trajectory prompt differs from the training record")
    with np.load(filename.parent / meta["chunks"][chunk_id], allow_pickle=False) as data:
        images = {}
        for role in ("image", "wrist_image"):
            image = data[role][local_index]
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Expected a uint8 RGB camera frame for {role}")
            images[f"observation/{role}"] = image
        state = data["state"][local_index]
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected xyz, axis-angle and gripper positions")
        return {**images, "observation/state": state, "actions": action, "prompt": row["goal"]}


class LiberoTraining:
    def __init__(self, settings):
        self.device = torch.device(settings.get("device", "cuda"))
        prefixes = settings.get("trainable_prefixes", (
            "paligemma_with_expert.gemma_expert.", "action_in_proj.", "action_out_proj.",
            "time_mlp_in.", "time_mlp_out."))
        if not isinstance(prefixes, (tuple, list)) or not prefixes or not all(isinstance(p, str) for p in prefixes):
            raise ValueError("trainable_prefixes must be a nonempty list of parameter prefixes")
        self.prefixes = tuple(prefixes)
        self.train_config = settings.get("model_config", "pi05_libero_v8_lora")

    def load_policy(self, checkpoint):
        from openpi import transforms
        from openpi.training import config
        from openpi.shared import normalize

        self.initial = Path(checkpoint).resolve()
        entry = config.get_config(self.train_config)
        info = json.loads((self.initial / "config.json").read_text())
        if not entry.model.pi05 or (info["action_dim"], info["action_horizon"]) != (32,10):
            raise ValueError("Expected the dense LIBERO π0.5 export with 10-step action targets")
        dense = replace(entry.model, paligemma_variant=info["paligemma_variant"],
                        action_expert_variant=info["action_expert_variant"],
                        dtype="float32", pytorch_compile_mode=None)
        cfg = replace(entry, model=dense)
        network = dense.load_pytorch(cfg, str(self.initial / "model.safetensors"))
        for key, p in network.named_parameters():
            p.requires_grad_(key.startswith(self.prefixes))
        schema = cfg.data.create(cfg.assets_dirs, dense)
        stats = normalize.load(self.initial / "assets" / schema.asset_id)
        if not stats:
            raise ValueError("LIBERO policy normalization assets required")
        self.transform = transforms.compose([
            *schema.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=schema.use_quantile_norm),
            *schema.model_transforms.inputs,
        ])
        return network.to(self.device)

    def _stack(self, rows, masks):
        import jax
        from openpi.models.model import Observation

        packed = jax.tree.map(lambda *values: torch.as_tensor(np.stack(values), device=self.device),
                              *(self.transform(entry) for entry in rows))
        actions = packed.pop("actions").float()
        obs = Observation.from_dict(packed)
        valid = torch.tensor(masks, dtype=torch.float32, device=self.device)
        return obs, actions, valid

    def supervised_batch(self, rows, rng):
        samples, masks = [], []
        for row in rows:
            _, _, all_actions = _episode(row["trajectory"])
            start = rng.randrange(len(all_actions))
            stop = min(start+10, len(all_actions))
            actions = np.asarray(all_actions[start:stop], dtype=np.float32)
            missing = 10-len(actions)
            samples.append(_frame(row, start, np.pad(actions, ((0, missing), (0, 0)), mode="edge")))
            masks.append([1] * len(actions) + [0] * missing)
        return self._stack(samples, masks)

    def preference_batch(self, rows, rng):
        positive, negative = [], []
        for row in rows:
            if row["action_horizon"] != 10 or row["control_interface"] != "libero_osc_pose_7d":
                raise ValueError("Unexpected preference action convention")
            observation = {"trajectory": row["positive_trajectory"], "goal": row["goal"]}
            for role, group in (("positive", positive), ("negative", negative)):
                action = np.load(row[role], allow_pickle=False)
                if action.shape != (10, 7) or action_digest(action) != row[f"{role}_sha256"]:
                    raise ValueError(f"Executed {role} action target changed after admission")
                if role == "positive":
                    _, _, recorded = _episode(row["positive_trajectory"])
                    if action_digest(recorded[:10]) != row["positive_sha256"]:
                        raise ValueError("Chosen target does not begin at the correction's recorded observation")
                group.append(_frame(observation, 0, action))
        masks = [[1] * 10 for _ in rows]
        positive = self._stack(positive, masks)
        negative = self._stack(negative, masks)
        return positive, negative, rng.getrandbits(62)

    def supervised_losses(self, model, batch):
        obs, actions, valid = batch
        losses = model(obs, actions).float().mean(dim=-1)
        return (losses * valid).sum(dim=-1) / valid.sum(dim=-1)

    def preference_losses(self, model, batch):
        positive, negative, seed = batch
        device = next(model.parameters()).device
        scores = []
        for target in (positive, negative):
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                scores.append(self.supervised_losses(model, target))
        return tuple(scores)

    def save_policy(self, model, directory):
        from safetensors.torch import save_model

        save_model(model, str(directory / "model.safetensors"))
        shutil.copy2(self.initial / "config.json", directory / "config.json")
        shutil.copytree(self.initial / "assets", directory / "assets")


def build(settings):
    return LiberoTraining(settings)
