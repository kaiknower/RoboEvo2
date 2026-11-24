"""Disjoint native task shards and strict journal merging; no runtime imports."""
from pathlib import Path
import json
from .campaign import GROUPS, CONDITIONS, summarize


def shard_keys(config, shard_id, shard_count=5):
    tasks = config['task_ids']
    if type(shard_count) is not int or not 1 <= shard_count <= len(tasks):
        raise ValueError('Invalid shard count')
    if type(shard_id) is not int or not 0 <= shard_id < shard_count:
        raise ValueError('Invalid shard identifier')
    start, end = len(tasks)*shard_id//shard_count, len(tasks)*(shard_id+1)//shard_count
    return [(t, i) for t in tasks[start:end] for i in config['init_ids']]


def plan_jobs(config, shard_count=5):
    return [dict(group=g, condition=c, shard_id=s, shard_count=shard_count,
                 case_keys=shard_keys(config, s, shard_count))
            for s in range(shard_count) for c in CONDITIONS for g in GROUPS]


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def merge_condition(directory, group, condition):
    directory = Path(directory)
    rows = read_rows(directory/'episodes.jsonl')
    journals = sorted((directory/'shards').glob('shard-*/episodes.jsonl'))
    plan = json.loads((directory/'shards'/'plan.json').read_text()) if journals else None
    for journal in journals:
        shard_id = str(int(journal.parent.name.split('-')[-1]))
        selected = {tuple(k) for k in plan['case_keys'][shard_id]}
        shard_rows = read_rows(journal)
        if any((row['task_id'], row['init_id']) not in selected for row in shard_rows):
            raise ValueError('Outcome is outside its assigned shard')
        rows.extend(shard_rows)
    return rows, summarize(rows, group, condition)
