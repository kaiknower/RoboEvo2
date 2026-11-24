import json
from pathlib import Path
import shutil
from .runtime import setup_runtime
from .campaign import Condition


def export_checkpoint(condition, model_path, baseline, output):
    Condition.parse(condition)
    model_path, baseline, output = map(lambda p: Path(p).resolve(), (model_path, baseline, output))
    if output.exists():
        raise FileExistsError(output)
    candidates = [model_path / "actor/model_state_dict/full_weights.pt", model_path / "model_state_dict/full_weights.pt"]
    weights = next((p for p in candidates if p.is_file()), None)
    if weights is None:
        if model_path != baseline or not (baseline / "model.safetensors").is_file():
            raise FileNotFoundError(f"No full_weights.pt in {model_path}")
        shutil.copytree(baseline, output)
        status = "baseline_retained"
    else:
        setup_runtime(output.parent / "export-runtime")
        import torch
        from safetensors.torch import save_file
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected RLinf full model state dictionary")
        output.mkdir(parents=True)
        save_file({k:v.contiguous().clone() for k,v in state.items()}, str(output / "model.safetensors"))
        shutil.copytree(baseline / "assets", output / "assets")
        for name in ("config.json", "conversion.json"):
            shutil.copy2(baseline / name, output / name)
        status = "ppo_checkpoint"
    (output / "policy.json").write_text(json.dumps(dict(condition=condition, source=str(model_path),
        baseline=str(baseline), status=status, algorithm="flow_sde_ppo"), indent=2) + "\n")
    return output
