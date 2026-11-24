#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${LIBERO_PRO_VENDOR_ROOT:?Set LIBERO_PRO_VENDOR_ROOT to the vendor directory in the pinned LIBERO-PRO checkout.}"
VENDOR=$(cd "$LIBERO_PRO_VENDOR_ROOT" && pwd -P)
REVISION=93012e94349084cc09b9bde675f603afadb4e04d
if [[ ! -f "$VENDOR/openpi/uv.lock" || $(git -C "$VENDOR/.." rev-parse HEAD) != "$REVISION" ]]; then
  echo "LIBERO_PRO_VENDOR_ROOT must point to vendor/ in the LIBERO-PRO checkout at $REVISION." >&2
  exit 1
fi
export LIBERO_PRO_VENDOR_ROOT="$VENDOR"
cd "$ROOT"
command -v uv >/dev/null
command -v zstd >/dev/null || { echo "Install zstd first (e.g. apt install zstd)." >&2; exit 1; }
UV_PROJECT_ENVIRONMENT="$ROOT/.venv" uv sync --project "$VENDOR/openpi" --python 3.11 --frozen --no-dev
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" -e "$VENDOR/rlinf" -e '.[test]' \
  'torch==2.7.1' 'transformers==4.53.2' 'numpy==1.26.4' 'ray[default]==2.48.0' \
  'gym==0.25.2' 'gymnasium==0.29.1' 'robosuite==1.4.1' 'mujoco==3.2.7' \
  'bddl==1.0.1' future matplotlib easydict cloudpickle 'setuptools<81' 'scipy<1.15'
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
  --reinstall-package torch --reinstall-package torchvision \
  'torch==2.7.1' 'torchvision==0.22.1'
uv pip install --python "$PY" --no-deps -e "$VENDOR/libero"
"$PY" - <<'PY'
import pathlib, shutil, transformers
from libero_pro.runtime import VENDOR_ROOT

source = VENDOR_ROOT / 'openpi/src/openpi/models_pytorch/transformers_replace'
shutil.copytree(source, pathlib.Path(transformers.__file__).parent, dirs_exist_ok=True)
PY
"$PY" scripts/fetch_assets.py
echo "Activate: source $ROOT/.venv/bin/activate"
echo "Next: python -m libero_pro download && python -m libero_pro convert"
