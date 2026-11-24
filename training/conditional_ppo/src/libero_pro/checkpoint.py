"""Restore the exact v8 baseline from its published archive."""
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ARCHIVE_PREFIX = "archives/checkpoints/pi05_libero_v8_lora/"


def extract_parts(parts, destination):
    """Stream split zstd tar parts; never materialize another concatenated archive."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".v8-extract-", dir=destination.parent))
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-dc"], stdin=cat.stdout, stdout=subprocess.PIPE)
    cat.stdout.close()
    try:
        with tarfile.open(fileobj=zstd.stdout, mode="r|") as archive:
            archive.extractall(temp, filter="data")
        if zstd.wait() != 0 or cat.wait() != 0:
            raise RuntimeError("Archive decompression failed")
        matches = list(temp.glob("**/v8_lora/29999/params/_METADATA"))
        if len(matches) != 1:
            raise ValueError(f"Expected one v8 step 29999, found {len(matches)}")
        shutil.move(str(matches[0].parent.parent), destination)
    finally:
        for proc in (cat, zstd):
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
        shutil.rmtree(temp)
    return destination


def download_baseline(repo, revision, destination):
    from huggingface_hub import snapshot_download
    if not revision:
        raise ValueError("Pin the baseline revision from configs/baseline.json")
    local = Path(snapshot_download(repo, revision=revision, allow_patterns=[f"{ARCHIVE_PREFIX}*", "norm_stats.json"]))
    parts = sorted((local / ARCHIVE_PREFIX).glob("*.part-*"))
    if len(parts) != 2:
        raise ValueError("Missing v8 archive parts")
    step = extract_parts(parts, destination)
    norm_dest = step / "assets/roboevo2/libero_expert_v8/norm_stats.json"
    norm_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local / "norm_stats.json", norm_dest)
    return step
