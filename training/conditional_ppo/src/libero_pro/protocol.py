"""Shared train/eval protocol checks, independent of simulator packages."""
import re
from pathlib import Path


def bddl_language(path):
    text = Path(path).read_text()
    match = re.search(r"\(:language\s+([^\)]+)\)", text)
    if not match or not match.group(1).strip():
        raise ValueError(f"No BDDL language in {path}; refusing filename fallback")
    return match.group(1).strip()


def reset_pool(trial_counts, trial_ids, task_ids=None):
    if not trial_ids or len(set(trial_ids)) != len(trial_ids) or any(type(i) is not int or i < 0 for i in trial_ids):
        raise ValueError("Trial IDs must be unique nonnegative integers")
    tasks = list(range(len(trial_counts))) if task_ids is None else list(task_ids)
    if not tasks or any(type(i) is not int or not 0 <= i < len(trial_counts) for i in tasks):
        raise ValueError("Invalid task IDs")
    offsets = [sum(trial_counts[:t]) for t in range(len(trial_counts))]
    if any(max(trial_ids) >= trial_counts[t] for t in tasks):
        raise ValueError("A requested init state is absent; refusing silent truncation")
    return [offsets[t] + i for i in trial_ids for t in tasks]


def environment_specs(task_suite, task_ids, bddl_root, base_params, seed, env_idx=None):
    paths = []
    descriptions = []
    for tid in task_ids:
        task = task_suite.get_task(tid)
        path = Path(bddl_root) / task.problem_folder / task.bddl_file
        paths.append(path)
        descriptions.append(bddl_language(path))
    indices = range(len(task_ids)) if env_idx is None else env_idx
    params = [dict(base_params, bddl_file_name=str(paths[i]), seed=seed) for i in indices]
    return params, descriptions
