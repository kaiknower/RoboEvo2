"""CPU preparation by default. GPU execution requires an explicit command."""
from pathlib import Path
import argparse
import json
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'launch', 'worker', 'summarize'])
    parser.add_argument('--config', default=str(ROOT/'configs/campaign.local.json'))
    parser.add_argument('--output')
    parser.add_argument('--group', choices=['sft_v8', 'ppo'])
    parser.add_argument('--condition')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-for-idle', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Preserve completed outcomes and retry infrastructure interruptions')
    parser.add_argument('--adaptive-shards', action='store_true', help='Fill GPU/RAM capacity with disjoint task shards')
    parser.add_argument('--shard-id', type=int)
    parser.add_argument('--shard-count', type=int, default=5)
    parser.add_argument('--admission-seconds', type=float,
                        help='Optional adaptive launch window; already launched shards drain normally')
    args = parser.parse_args()
    if args.admission_seconds is not None and (args.command != 'launch' or not args.adaptive_shards):
        parser.error('--admission-seconds requires launch --adaptive-shards')
    from lharness.campaign import load_config, model_path, CONDITIONS, GROUPS, aggregate
    config = load_config(args.config)
    if args.command in {'prepare', 'preflight'}:
        rows = []
        for group in GROUPS:
            for condition in CONDITIONS:
                path = model_path(config, group, condition)
                model = path/'model.safetensors'
                stats = path/'assets/roboevo2/libero_expert_v8/norm_stats.json'
                rows.append(dict(group=group, condition=condition, model_path=str(path),
                    weights_present=model.is_file(), bytes=model.stat().st_size if model.is_file() else None,
                    normalization_present=stats.is_file(), episodes=100))
        ready = all(r['weights_present'] and r['normalization_present'] for r in rows)
        report = dict(kind='CPU_PREPARATION_ONLY', protocol=config['protocol'],
            native_episodes=0, gpu_started=False, background_waiter_started=False,
            models_present=ready, planned_episodes=1600, conditions=rows,
            cameras=config['cameras'], policy_prefix_caps=config['policy_caps'],
            recovery_actions=config['recovery_steps'], coding_model=config['coding_agent'].get('model'))
        (ROOT/'artifacts').mkdir(exist_ok=True)
        (ROOT/'artifacts/preflight.json').write_text(json.dumps(report, indent=2)+'\n')
        from lharness.scheduler import run
        queue = run(args.config, args.output or ROOT/'artifacts/queue_preview', execute=False,
                    adaptive_shards=args.adaptive_shards, shard_count=args.shard_count)
        print(json.dumps(dict(prepared=ready, jobs=len(queue['jobs']), planned_episodes=1600,
                             gpu_started=False, report=str(ROOT/'artifacts/preflight.json')), indent=2))
        return 0 if ready else 1
    output = args.output or config['output_root']
    if args.command == 'launch':
        from lharness.scheduler import run
        result = run(args.config, output=output, execute=args.execute,
                     wait_for_idle=args.wait_for_idle, once=args.once, resume=args.resume,
                     adaptive_shards=args.adaptive_shards, shard_count=args.shard_count,
                     admission_seconds=args.admission_seconds)
        if args.execute:
            report = aggregate(output)
            (Path(output)/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status': result['status'], 'output': output}, indent=2))
        return 0 if result['status'] in {'prepared', 'finished'} else 1
    if args.command == 'worker':
        if args.group is None or args.condition not in CONDITIONS:
            parser.error('worker requires --group and a configured --condition')
        from lharness.evaluation import run_condition
        report = run_condition(config, args.group, args.condition, output, resume=args.resume,
                               shard_id=args.shard_id, shard_count=args.shard_count)
        return 0 if report['status'] == 'complete' else 1
    report = aggregate(output)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
