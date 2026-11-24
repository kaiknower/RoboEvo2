# LIBERO-PRO v8 supervised and preference training

Run `train.py` with the patched OpenPI implementation in the companion [LIBERO-PRO runtime](https://github.com/YOUR_ORG/libero-pro/tree/93012e94349084cc09b9bde675f603afadb4e04d). The launcher checks the source hashes of the trainer, configs and data and preference modules. It uses the OpenPI JAX optimizer and loss, records resolved inputs and metrics, and saves complete Orbax checkpoints with normalization assets.

| Recipe | Initialization | Optimizer updates | Samples |
|---|---|---:|---|
| `original-sft` | Original π0.5-LIBERO JAX parameters | 30,000 | 960,000 demonstrations at batch 32 |
| `sft-only` | Merged SFT v8 JAX parameters | 10,000 at batch 96, or 12,000 at batch 80 | 960,000 demonstrations |
| `sft-cpd` | Merged SFT v8 JAX parameters | 30,000 at fixed batch, or 24,667 in the two-phase schedule | 960,000 demonstrations and 240,000 pairs |
| `sft-cpd-no-prox` | Merged SFT v8 JAX parameters | 20,000 | 960,000 demonstrations at batch 48 and 240,000 pairs at batch 12 |

`original-sft` selects the `pi05_libero_v8_lora` recipe, which trains dense weights along with its rank-16/rank-32 LoRA factors. The three continuations use the rank-512 LoRA mask. The pure SFT continuation uses the native BC branch and creates only the BC loader. Both CPD continuations use IPO with β 0.5 and preference weight 0.02. Their fixed reference is the merged v8 policy with the newly trained adapters disabled. The proximal variant uses the one-sided penalty with weight 0.5 and slack 0.3. All four recipes have EWC disabled.

## Environment and local inputs

Use Linux, Python 3.11, CUDA-compatible NVIDIA GPUs for training, and the exact companion OpenPI checkout. Install its locked dependencies in an OpenPI-specific environment.

```sh
export OPENPI_ROOT=/path/to/libero-pro/vendor/openpi
uv sync --project "$OPENPI_ROOT" --python 3.11 --frozen --no-dev
export OPENPI_PYTHON="$OPENPI_ROOT/.venv/bin/python"
```

The lock pins JAX 0.5.3, Flax 0.10.2, Orbax 0.11.13 and LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. Use that LeRobot with datasets 4.8.5; the launcher includes the matching numeric Arrow-column compatibility code. OpenPI caches the PaliGemma tokenizer at `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` or under `~/.cache/openpi` if that variable is unset. Populate that file ahead of time for an offline training run.

Supply the benchmark inputs from your own authorized copies.

- `--data-root` points to the LeRobot v2.1 root containing `roboevo2/libero_expert_v8`. The dataset has 6,593 demonstrations and 1,119,092 frames. The two CPD recipes also need `roboevo2/libero_cpd_strong_pairs_v8champ`, with 701 one-frame episodes and `pairs_manifest.json`. Each dataset needs its `meta/` and `data/` trees.
- `--assets` points to a directory containing `roboevo2/libero_expert_v8/norm_stats.json`. The normalization file is also available in the released v8 checkpoint’s `assets/` directory.
- `--params` points directly to an Orbax JAX `params` directory. For original SFT, use the original π0.5-LIBERO initialization from OpenPI, `gs://openpi-assets/checkpoints/pi05_libero/params`. For the continuations, merge the released SFT v8 JAX checkpoint with the companion `scripts/merge_lora_into_base.py`. Given its checkpoint root, use `--src "$V8_CHECKPOINT/params" --out "$MERGED_V8/params"`.

## Launch

From this directory, inspect the recipe and validate input paths and metadata on CPU.

```sh
python train.py sft-cpd --phase continuation --plan
"$OPENPI_PYTHON" train.py sft-cpd --check-only --openpi-root "$OPENPI_ROOT" \
  --params "$MERGED_V8/params" --assets "$V8_CHECKPOINT/assets" --data-root "$LEROBOT_ROOT"
```

Training uses the GPUs selected by `CUDA_VISIBLE_DEVICES`. The BC and preference batch sizes must each divide evenly across the visible devices. The CPD initial phase expects four visible devices with FSDP 4, and the continuation and no-prox recipes six visible devices with FSDP 1. Select a different compatible sharding with `--fsdp-devices` when needed.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 "$OPENPI_PYTHON" train.py original-sft \
  --openpi-root "$OPENPI_ROOT" --params "$PI05_LIBERO/params" \
  --assets "$V8_CHECKPOINT/assets" --data-root "$LEROBOT_ROOT" --output "$OUTPUT_ROOT/original-sft"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$OPENPI_PYTHON" train.py sft-only \
  --openpi-root "$OPENPI_ROOT" --params "$MERGED_V8/params" \
  --assets "$V8_CHECKPOINT/assets" --data-root "$LEROBOT_ROOT" --output "$OUTPUT_ROOT/sft-only"
```

Use `sft-cpd-no-prox` with the same merged inputs and its own output directory. For the two-phase proximal CPD schedule, start `sft-cpd --phase initial` with four devices and a new output directory. It stops after update 14,001 and writes checkpoint `checkpoints/14000`. Then launch `sft-cpd --phase continuation --from-checkpoint "$INITIAL_OUTPUT/checkpoints/14000"` with six devices and a new output directory. The continuation restores optimizer state into the new device layout, advances both seeded samplers past the first phase, and reaches checkpoint `24666` after update 24,667. The fixed batch recipe is also available with `sft-cpd` and the default `--phase full`.

For an interrupted run, keep the same inputs, output directory and phase and add `--resume`. For the continuation, omit `--from-checkpoint` when resuming its own saved state. The launcher validates its recorded inputs before restoring. Checkpoint names are zero-based and parameters live under `checkpoints/<index>/params`.

Training ends at checkpoint publication. Use the repository’s policy evaluation entry points with a converted export and the held split to run a new evaluation.

## Joint SFT, CPD and EWC

[`train_cpd_ewc.py`](train_cpd_ewc.py) adds the factorized effective-weight EWC penalty inside the joint CPD loss, before differentiation. It preserves the CPD preference weight of 0.02 and proximal weight of 0.5. The EWC coefficient is `4e-6`, with a factor of one half in the objective.

Use the patched OpenPI tree from the companion LIBERO-PRO checkout as `OPENPI_ROOT`. The launcher verifies its relevant source hashes. Use Python 3.11, JAX 0.5.3, Flax 0.10.2, Orbax 0.11.13, the pinned LeRobot revision above, datasets 4.8.5 and PyTorch 2.10.0+cu128.

Set `INIT_PARAMS` to either the official `pi05_base/params` or the merged SFT v8 `params` directory. Compute a separate [factorized Fisher](../libero_ewc/README.md#factorized-fisher-for-cpd--ewc) for each initializer and point `FISHER` to its output NPZ. Use the same normalization assets and LeRobot data for both variants. Set `OUTPUT_ROOT` to a different directory for each variant.

The first 2,000 optimizer updates use 32 BC examples and 8 preference pairs per update. The continuation uses 704 and 176, with a final partial batch of 512 and 128. This yields 3,273 updates and preserves the budget of 960,000 BC draws and 240,000 preference-pair draws.

From this directory, the following Bash commands first preview the budget and validate the local inputs on CPU.

```bash
python3 train_cpd_ewc.py sft-cpd --phase larger-batch --first-update 2000 --batch-size 704 --plan

common=(sft-cpd --openpi-root "$OPENPI_ROOT" --params "$INIT_PARAMS"
  --assets "$V8_CHECKPOINT/assets" --data-root "$LEROBOT_ROOT"
  --fisher "$FISHER" --fsdp-devices 4)
CUDA_VISIBLE_DEVICES='' "$OPENPI_PYTHON" train_cpd_ewc.py "${common[@]}" --check-only --stop-after 2000
```

Run the two phases in order on four dedicated GPUs. The 704-example continuation needs four 80 GiB GPUs with a 0.96 XLA device memory fraction.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$OPENPI_PYTHON" train_cpd_ewc.py "${common[@]}" --stop-after 2000 --output "$OUTPUT_ROOT/initial"

XLA_CLIENT_MEM_FRACTION=0.96 XLA_PYTHON_CLIENT_PREALLOCATE=true \
  "$OPENPI_PYTHON" train_cpd_ewc.py "${common[@]}" --phase larger-batch --first-update 2000 \
  --batch-size 704 --from-checkpoint "$OUTPUT_ROOT/initial/checkpoints/1999" \
  --output "$OUTPUT_ROOT/continuation"
```

The final checkpoint is `continuation/checkpoints/3272`. To resume an interrupted phase, keep its inputs and output directory, add `--resume`, and omit `--from-checkpoint`. For a bounded stop followed by continuation of the same phase, remove `--stop-after` when resuming.

## Source

The trainer and configuration are in the patched OpenPI tree of the companion checkout. The [CPD objective](reference_source/pi0_pref.py) is included for reference; training imports the companion module after checking its hash. The included OpenPI license covers that file.
