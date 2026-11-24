# LIBERO-PRO conditional PPO

This directory contains the eight-condition pi05 v8 training and validation code. Each condition starts from the same SFT checkpoint and runs independent PPO with Flow-SDE. Rollout and updates use the full three-transition denoising likelihood. The training split uses initial-state IDs 0–9, validation uses 20–29, and held-out tests use 10–19. Validation measures the SFT starting point and every subsequent iteration. The campaign exports the validation-selected policy for each condition, then evaluates both SFT and the selected policy on the held-out test split.

## Runtime setup

Use Linux, Python 3.11, an NVIDIA GPU driver compatible with CUDA 12.8, and the EGL/OpenGL system libraries required by MuJoCo. Install `git`, `uv`, and `zstd`. The companion checkout provides the patched RLinf, openpi and LIBERO implementations, its OpenPI dependency lock, and the benchmark's definitions and initial states. Use the pinned checkout listed in [THIRD_PARTY.md](THIRD_PARTY.md).

From this directory, place the companion checkout outside the release tree and run setup. `setup.sh` creates `.venv` here and fetches the benchmark meshes into the companion checkout when absent.

```bash
git clone https://github.com/YOUR_ORG/libero-pro.git ../../../libero-pro-runtime
git -C ../../../libero-pro-runtime checkout --detach 93012e94349084cc09b9bde675f603afadb4e04d
export LIBERO_PRO_VENDOR_ROOT="$(cd ../../../libero-pro-runtime/vendor && pwd)"
bash scripts/setup.sh
source .venv/bin/activate
```

Keep `LIBERO_PRO_VENDOR_ROOT` set when invoking the training code or using it as the Hybrid evaluator's runtime. For Hybrid, set `LIBERO_PRO_ROOT` to this directory and use its `.venv/bin/python`.

## Plan and preview

The plan command works from a source checkout using only the Python standard library. The training preview additionally composes the real RLinf Hydra config from the companion checkout and reports its path.

```bash
PYTHONPATH=src python -m libero_pro plan --output plan.json
CUDA_VISIBLE_DEVICES='' python -m libero_pro train --condition spatial_position \
  --model weights/v8-pytorch --output-root preview --dry-run
```

## Start from the v8 baseline

Download the v8 baseline checkpoint from `YOUR_HF_ORG/libero-pro` (a placeholder until the checkpoint is released). The download command uses the revision in `configs/baseline.json`. For an already converted baseline, pass its directory as `--model`. It must contain `model.safetensors`, `conversion.json`, and `assets/roboevo2/libero_expert_v8/norm_stats.json`.

```bash
hf auth login
python -m libero_pro download --output weights/v8-original
JAX_PLATFORMS=cpu python -m libero_pro convert \
  --input weights/v8-original --output weights/v8-pytorch
```

## Run training

The default recipe uses eight GPUs, 200 iterations per condition, global batch size 2048 and per-GPU microbatch size 128. Use `train --condition spatial_position` for a single condition or `run-eight` for the sequential campaign.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m libero_pro run-eight \
  --model weights/v8-pytorch --output-root runs \
  --iterations 200 --micro-batch-size 128
```

Each condition saves validation reports and a validation curve under `runs/<condition>/`. The campaign writes selected policies to `runs/final/<condition>/` and held-out aggregates to `runs/comparison.json`.
