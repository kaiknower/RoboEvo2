"""Exact suite-axis assets, separate trial pools and binary first-success reward."""
import os
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from rlinf.envs.libero.libero_env import LiberoEnv
from .protocol import environment_specs, reset_pool


class LiberoProEnv(LiberoEnv):
    def _compute_total_num_group_envs(self):
        if os.environ.get("LIBERO_INIT_ID_ALLOWLIST") or self.specific_reset_id is not None:
            raise ValueError("Use cfg.trial_ids; global allowlists/specific_reset_id bypass split isolation")
        super()._compute_total_num_group_envs()
        self._valid_reset_state_ids = np.asarray(reset_pool(
            self.trial_id_bins, list(self.cfg.trial_ids), self.task_id_filter), dtype=np.int64)

    def get_env_fn_params(self, env_idx=None):
        from libero.libero import get_libero_path
        root = Path(get_libero_path("bddl_files"))
        base = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        params, descriptions = environment_specs(
            self.task_suite, self.task_ids, root, base, self.seed, env_idx)
        self.task_descriptions = descriptions
        return params

    def _calc_step_reward(self, terminations):
        success = np.asarray(terminations, dtype=bool)
        reward = success & ~self.prev_step_reward.astype(bool)
        self.prev_step_reward = np.logical_or(self.prev_step_reward, success)
        return reward.astype(np.float32)
