# LIBERO-PRO Hybrid: targeted repair

This runner revisits policy failures recorded by the [Hybrid evaluator](../full_campaign/). For each failure it replays the saved policy actions, checks that the replay reproduces the recorded RGB-D and robot state, and then lets the harness attempt a bounded recovery from that point. It contains the public controller and perception checks, the policy adapter, and its own skills and memory.

Use a Linux Python 3.11 environment with the companion [LIBERO-PRO runtime](https://github.com/YOUR_ORG/libero-pro), its modified LIBERO/Robosuite, RLinf and openpi dependencies, PyTorch, NumPy, and OpenCV. Set `LIBERO_PRO_ROOT` to the companion checkout if it is not already importable. The SFT v8 and per-condition PPO checkpoints are obtained separately.

1. Implement `query_model()` in [`lharness/model_api.py`](lharness/model_api.py) for your model, as described in [Connect your model](../../README.md#connect-your-model). This directory has its own copy of that file.
2. Copy `config.example.json` to a local file. Set all absolute paths and GPU IDs, including the runtime Python interpreter, and keep the new output root separate from the read-only baseline output.
3. Build a repair manifest from completed native failures of your own Hybrid run. Each case needs its `outcome.json`, `actions_proprioception.npz` and handoff RGB images under `coding/request_01/`.

```bash
python make_targets.py --config /path/to/config.json --output /path/to/targets.json /path/to/baseline-case
python run.py prepare --config /path/to/config.json --manifest /path/to/targets.json --round round_01
python run.py launch --config /path/to/config.json --manifest /path/to/targets.json --round round_01 --max-live 1 --execute
python verify_repair_round.py /path/to/output/round_01
```

`prepare` only prints the dispatch plan. The `--execute` line starts simulation and calls your model. The verifier separates infrastructure errors from completed native failures.

Optional segmentation proposals for drawer and tabletop handles come from the [SAM3 service](../perception/README.md). To use them, set `coding_agent.public_visual_proposals.enabled=true`, its `endpoint` to the service's `/segment` URL, `tabletop_enabled=true`, and a shared absolute `admission_root`. The planner treats proposals as hints; the support-surface geometry guard and sensor filtering run in this code, and native success stays outside the planner's context.
