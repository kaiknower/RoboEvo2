# Source and dependency notices

The training code runs against [`YOUR_ORG/libero-pro` at `93012e94349084cc09b9bde675f603afadb4e04d`](https://github.com/YOUR_ORG/libero-pro/tree/93012e94349084cc09b9bde675f603afadb4e04d). Its patched dependencies are resolved from that checkout through `LIBERO_PRO_VENDOR_ROOT`.

| Directory in the companion checkout | Upstream | License file |
|---|---|---|
| `vendor/rlinf` | [RLinf](https://github.com/RLinf/RLinf) | `vendor/rlinf/LICENSE` (Apache-2.0) |
| `vendor/openpi` | [openpi](https://github.com/Physical-Intelligence/openpi) | `vendor/openpi/LICENSE`, `vendor/openpi/LICENSE_GEMMA.txt` |
| `vendor/libero` | [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) | `vendor/libero/LICENSE` |

The patched RLinf supplies exact condition selection, config-local split pools, ten-step settling, strict checkpoint loading, full denoising-path likelihood sums, finite-cap failure handling, pi05 conversion and synchronous per-iteration validation using the current rollout policy. It uses openpi's normalization loader. The openpi patch removes the unused PyTorch action expert vocabulary output head to match the JAX parameters. The patched LIBERO loader sets `weights_only=False` for the benchmark's trusted initial states. The companion checkout retains the original notices.

`scripts/fetch_assets.py` fetches benchmark meshes from [the pinned LIBERO-PRO asset revision](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec). The task definitions and initial states are read from the companion checkout.

The gated v8 baseline in `configs/baseline.json` is an external input governed by its checkpoint terms.
