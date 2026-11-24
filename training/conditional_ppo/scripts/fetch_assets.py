"""Fetch meshes from the benchmark upstream; preserve vendored BDDL/init states."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from libero_pro.runtime import VENDOR_ROOT

REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"

if __name__ == "__main__":
    destination = VENDOR_ROOT / "libero/libero/libero/assets"
    if destination.is_dir() and any(destination.iterdir()):
        print(destination)
    else:
        with tempfile.TemporaryDirectory(prefix="libero-assets-") as tmp:
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/Zxy-MLlab/LIBERO-PRO.git", tmp], check=True)
            subprocess.run(["git", "-C", tmp, "checkout", REVISION, "--", "libero/libero/assets"], check=True)
            shutil.copytree(Path(tmp) / "libero/libero/assets", destination, dirs_exist_ok=True)
        print(destination)
