import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .runtime import ROOT, setup_runtime


def convert(source, output, merged_source=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    os.environ["JAX_PLATFORMS"] = "cpu"
    setup_runtime(output.parent / "conversion-runtime")
    merged = Path(merged_source).resolve() if merged_source else output.parent / (output.name + "-merged-jax")
    if merged_source:
        manifest = json.loads((merged / "MERGE_MANIFEST.json").read_text())
        if manifest["source"] != str(source / "params") or manifest["persisted_lora_leaves"] != 0:
            raise ValueError("The completed merge does not match this source")
    else:
        if merged.exists():
            raise FileExistsError(f"Use --merged {merged} to reuse its completed merge")
        subprocess.run([sys.executable, str(ROOT / "scripts/merge_lora_into_base.py"),
            "--src", str(source / "params"), "--out", str(merged / "params")], check=True)
    from openpi.training.config import get_config
    from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import convert_pi0_checkpoint
    original_config = get_config("pi05_libero_v8_lora").model
    dense_config = dataclasses.replace(original_config, paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m", dtype="float32", pytorch_compile_mode=None)
    convert_pi0_checkpoint(str(merged), "float32", str(output), dense_config)
    shutil.copytree(source / "assets", output / "assets", dirs_exist_ok=True)
    norm = output / "assets/roboevo2/libero_expert_v8/norm_stats.json"
    if not norm.is_file():
        raise FileNotFoundError(norm)
    report = dict(source=str(source), merged=str(merged), output=str(output),
        model_config=dataclasses.asdict(dense_config), norm_stats=str(norm.relative_to(output)),
        merge=json.loads((merged / "MERGE_MANIFEST.json").read_text()),
        converted=True, policy_success_remeasured=False)
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output)
