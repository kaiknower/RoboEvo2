# Start an evolution run

The round runner maintains two objects, a robot policy and a harness containing skills, working and episodic memory, and routing instructions. Round zero uses a demonstration-trained policy and a [small executable LIBERO skill catalog](initial/README.md). Its memory is empty. Later skills and experience are generated into the local run directory.

## One round

1. Run the frozen incoming policy on each training case. Store synchronized pre-action camera views, robot state, executed actions and the benchmark verdict.
2. For each failed case, create three branches from the same reset or verified replay anchor. **Code** uses public robot primitives, **Code+ICL** also sees a goal-matched training demonstration, and **Code+Policy** can call the frozen policy. Skills may be edited in a branch. Native success ends the attempt.
3. Among verified, comparable successes, choose the lowest measured normalized end-effector jerk. Accumulate the winning trajectory and direct policy successes as supervision. Admit a preference pair only for different executed action chunks at the same measured state and observation, with matching control interface, rate, horizon and execution conditions.
4. Integrate compatible procedures and general guidance. Test the candidate with the frozen policy on its original repaired cases and the training cases solved before integrating the edit. Store branch attempts and conflicts for inspection.
5. Train a candidate policy. Score both policies alone on the same development cases, then compare the two harnesses on those cases using the selected policy. Strict improvement promotes a candidate. A tie retains its parent. Repeat from the selected policy and harness.

The example uses training init IDs 0–9 and development IDs 20–29. Initial states 10–19 are reserved for the separate [held-out evaluator](../hybrid/README.md). The final benchmark is not an input to the update.

## Configure and run

Install the [pinned LIBERO-PRO runtime](../training/conditional_ppo/README.md#runtime-setup) and use its Python environment with NumPy, PyTorch, the patched OpenPI/RLinf and LIBERO packages, simulator assets, EGL and a compatible dense OpenPI PyTorch checkpoint. The planner calls your model through `query_model()` in [`hybrid/full_campaign/lharness/model_api.py`](../hybrid/full_campaign/lharness/model_api.py); see [Connect your model](../README.md#connect-your-model). The worker also accepts `--planner module:factory` for a different trusted planner.

```sh
cp evolution/config.example.json evolution/campaign.local.json
export LIBERO_PRO_ROOT=/absolute/path/to/companion/libero-pro
python -m evolution.run evolution/campaign.local.json --workspace runs/first --check
```

Edit the local copy before execution. Set `initial_policy` to a checkpoint directory containing `config.json`, `model.safetensors` and its original LIBERO normalization assets. Provide a demonstration for every training goal and use the exact native instruction in both the demo and case. Each training and development case specifies its native condition, task, initial state, seed and task length scale in metres. The 0.25 m scale and single training step in the example serve as configuration examples. Choose the scale, rollout cohorts and optimizer budget for the experiment.

Each D0 record names a training trajectory. The [capture format](libero_runtime.py) stores a JSON manifest, chunked NumPy arrays of pre-action policy-view RGB and wrist RGB, state and corresponding executed seven-dimensional actions, plus the complete executed action array. Verified demonstrations in that format can supply both Code+ICL context and supervision. Optional sampled RGB frames can provide ICL context when the aligned robot trajectory is available separately for training. The [OpenPI binding](libero_training.py) validates shapes, instruction, action convention and the identity of admitted preference targets.

```sh
python -m evolution.run evolution/campaign.local.json --workspace runs/first --rounds 1
python -m evolution.run evolution/campaign.local.json --workspace runs/first --rounds 3
```

`--rounds` is the total round count. The second command continues to round three. The runner locks the workspace, reuses completed stage responses and keeps interrupted attempts for diagnosis. Keep input assets and worker implementations fixed when resuming. Changed assets, code or hyperparameters belong in a fresh workspace.

## Policy update

The [training driver](policy_update.py) computes per-example supervised losses, a preference gap and a proximal penalty against the frozen policy at the start of the round, and an EWC penalty weighted by empirical Fisher information from the prior demonstration pool. It samples demonstrations and preference pairs independently, using a half-uniform and half-failure-weighted task distribution. When the pool contains no eligible preference pairs, supervision and EWC remain active.

The [LIBERO π0.5 adapter](libero_training.py) supplies native flow-matching losses, matched observation/action chunks and a reloadable dense checkpoint with normalization assets. By default it trains the action expert and projection/time heads. Use `python -m evolution.policy_update --help` for optimizer settings and the native adapter interface.

## Worker protocol

The five entries in `commands` are argument arrays. A leading `{python}` resolves to the current interpreter. The driver appends an input JSON path and an output JSON path. The native adapters implement rollout, repair, evaluation, integration and training. Every attempt gets its own directory. Stage errors and unjudged native episodes interrupt the round and remain available for retry. The selected policy, harness and development counts are written to `round-NNNN/selected.json`.

To check the example schema alone, the `--check` command needs only the Python standard library.
