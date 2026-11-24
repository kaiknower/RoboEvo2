# LIBERO v8 EWC training

`run.py` launches the LIBERO v8 π0.5 elastic weight consolidation recipes. `openpi_train.py` contains the training step and Fisher penalties. The two Fisher scripts compute the corresponding weight estimates on task-A data.

| Recipe | Objective and inputs |
| --- | --- |
| `pi05_libero_v8_ewc_lora` | Behavior cloning with EWC on the effective LoRA weight update. Uses the row/column-factorized Fisher NPZ and defaults to λ = 4e-6. |
| `pi05_libero_v8_ewc_r256_lora` | A rank-256 LoRA variant. Accepts the same Fisher NPZ and defaults to λ = 4e-7. |
| `pi05_libero_v8_paper_ewc_full` | Dense behavior cloning with an exact diagonal empirical Fisher estimated from individual task-A examples. Anchors all trainable weights at the task-A checkpoint. The config sets λ = 1e-8, and `--ewc-lambda` sets another value. |
| `pi05_libero_v8_paper_ewc_full_control` | The dense behavior-cloning control, using the same task-B episode filter and initialization as the dense EWC recipe. |

The LoRA estimator squares batch-mean gradients, factors the base-weight diagonal into row and column sums, and normalizes its global mean to one. The dense estimator averages squared per-example gradients without normalizing the stored diagonal. Training multiplies the Fisher-weighted squared displacement by λ/2 and adds it to the flow-matching behavior-cloning loss. The dense and LoRA Fisher files use different formats and normalization assets.

## Setup

Use the OpenPI environment for the [pinned companion LIBERO-PRO checkout](https://github.com/YOUR_ORG/libero-pro/tree/93012e94349084cc09b9bde675f603afadb4e04d). The [v8 training setup](../libero_v8/README.md#environment-and-local-inputs) describes the Python version, dependency lock, and tokenizer cache. Training and Fisher computation require CUDA GPUs with enough memory for the selected recipe.

Supply the original π0.5-LIBERO base checkpoint's `params/` as `--params`. `--data-root` points to a local LeRobot v2.1 data home containing the datasets at their repo-relative paths. The default task-A dataset is `physical-intelligence/libero` for Fisher computation. The default training dataset is `roboevo2/libero_expert_v8`. Use `--repo-id` to select a local dataset ID.

For LoRA, `--assets` must contain `roboevo2/libero_expert_v8/norm_stats.json`. For dense runs, it must contain `physical-intelligence/libero/norm_stats.json` from the task-A checkpoint assets. Use the same asset directory and task-A initialization for a given Fisher computation and its training run. The dense training recipes also take `--episode-filter`, the task-B episode allowlist JSON. Pass the task-B episode allowlist. OpenPI accepts a sorted array of episode IDs or an object with `episodes` and an optional matching `repo_id`.

## Commands

From the release root, set `OPENPI_PYTHON` and `OPENPI_ROOT` as in the v8 setup and set the input and output variables to your local directories. If you already have the matching Fisher file, start with the training command.

```sh
"$OPENPI_PYTHON" training/libero_ewc/run.py fisher pi05_libero_v8_ewc_lora \
  --openpi-root "$OPENPI_ROOT" --data-root "$DATA_ROOT" \
  --params "$TASK_A_PARAMS" --assets "$V8_ASSETS" --out "$OUTPUT_ROOT/ewc_fisher.npz"

"$OPENPI_PYTHON" training/libero_ewc/run.py train pi05_libero_v8_ewc_lora \
  --openpi-root "$OPENPI_ROOT" --data-root "$DATA_ROOT" \
  --params "$TASK_A_PARAMS" --assets "$V8_ASSETS" --fisher "$OUTPUT_ROOT/ewc_fisher.npz" \
  --output-root "$OUTPUT_ROOT" --exp-name ewc-lora
```

For dense EWC, use `pi05_libero_v8_paper_ewc_full` in both commands, point `--assets` to the original task-A normalization assets, and use a Fisher path ending in `/params` in place of the NPZ. Add `--episode-filter "$TASK_B_EPISODES"` to the training command. Dense Fisher defaults to two visible devices and one example per gradient. Use the same `--fsdp-devices` as the number of visible devices when overriding its default. For the matched control, use the control recipe and omit `--fisher`.

Add `--check-only` to validate the resolved config and input paths on CPU without restoring Orbax weights. To resume an existing run, reuse its inputs, output root and experiment name and add `--resume`. These recipes add EWC to behavior cloning. The SFT and CPD continuation recipes are in [LIBERO v8 training](../libero_v8/README.md).

## Factorized Fisher for CPD + EWC

The combined trainer uses `fisher_shards.py` to estimate Fisher importance separately for the official π0.5 Base and merged SFT v8 initializations. Use the same initialization, LeRobot data root, normalization assets, and pinned OpenPI source as in the combined training run. Set `INIT_PARAMS` to one initialization's Orbax `params/` directory and `FISHER` to a fresh `.npz` path outside the checkout. Use a different output path for each initialization.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 "$OPENPI_PYTHON" training/libero_ewc/fisher_shards.py \
  fisher pi05_libero_v8_ewc_lora \
  --openpi-root "$OPENPI_ROOT" --data-root "$LEROBOT_ROOT" \
  --repo-id roboevo2/libero_expert_v8 --params "$INIT_PARAMS" --assets "$V8_CHECKPOINT/assets" \
  --steps 500 --batch-size 4 --out "$FISHER"
```

Append `--check-only` for an input check on CPU. A full run computes 500 shuffled batches of four demonstrations, with each GPU assigned a disjoint set of complete batches. `checked_fisher.py` uses precomputed seed-0 flow-matching draws and squares the batch-mean gradients at ten effective-weight sites. The merger checks that every batch was covered once, combines float64 row and column sums, and writes the normalized float32 Fisher factors used by the trainer.

The launcher retains the worker logs and intermediate shards next to the output. To merge already completed shards without recomputing them, pass all shard files using repeated `--shard` arguments to `merge_fisher_shards.py` and set `--batches 500` and `--output "$FISHER"`.

## Source

The trainer and named configs are in the patched OpenPI tree of the companion checkout (`scripts/train.py` and `src/openpi/training/config.py`), whose upstream notices apply.
