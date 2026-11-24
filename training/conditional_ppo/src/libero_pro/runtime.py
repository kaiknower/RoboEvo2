import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(os.environ.get("LIBERO_PRO_VENDOR_ROOT", ROOT / "vendor")).expanduser().resolve()
EMBODIED_PATH = VENDOR_ROOT / "rlinf/examples/embodiment"


def apply_cuda_memory_budget():
    """Optional per-policy-process allocator cap for shared-GPU smoke checks."""
    limit = os.environ.get("LIBERO_PPO_GPU_MEMORY_GIB")
    if limit is None:
        return
    import torch
    limit_bytes = float(limit) * 1024**3
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    if not 0 < limit_bytes <= total:
        raise ValueError("LIBERO_PPO_GPU_MEMORY_GIB must be within device capacity")
    torch.cuda.set_per_process_memory_fraction(limit_bytes / total, device)


def setup_runtime(run_dir):
    if not EMBODIED_PATH.is_dir():
        raise FileNotFoundError(
            f"Patched RLinf runtime not found at {EMBODIED_PATH}. "
            "Set LIBERO_PRO_VENDOR_ROOT to the vendor directory in the pinned LIBERO-PRO checkout."
        )
    run_dir = Path(run_dir).resolve()
    core = VENDOR_ROOT / "libero/libero/libero"
    config_dir = run_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(core), "bddl_files": str(core / "bddl_files"),
        "init_states": str(core / "init_files"), "assets": str(core / "assets"),
        "datasets": str(run_dir / "unused_datasets"),
    }))
    (run_dir / "unused_datasets").mkdir(exist_ok=True)
    paths = [ROOT / "src", VENDOR_ROOT / "rlinf", VENDOR_ROOT / "libero",
             VENDOR_ROOT / "openpi/src", VENDOR_ROOT / "openpi/packages/openpi-client/src"]
    for p in reversed(paths):
        sys.path.insert(0, str(p))
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["LIBERO_TYPE"] = "standard"
    os.environ["EMBODIED_PATH"] = str(EMBODIED_PATH)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return ROOT
