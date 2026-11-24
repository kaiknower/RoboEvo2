"""Create a disposable patched copy of the pinned OpenPI trainer.

The pinned OpenPI checkout is never written. Only the CPD loss_fn's
early return is extended with the factorized effective-weight EWC term.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


CPD_CONTEXT = "                        }\n                    )\n"
CPD_RETURN = CPD_CONTEXT + "            return loss.astype(jnp.float32), aux\n"
EWC_BLOCK = '''            if config.ewc_enabled:
                if ewc_fisher is None:
                    raise ValueError("ewc_enabled but no Fisher was threaded into CPD train_step.")
                trainable = traverse_util.flatten_dict(
                    nnx.state(model).filter(config.trainable_filter).to_pure_dict(), sep="/"
                )
                expected_sites = {path for path in trainable if path.endswith("lora_a")}
                fisher_sites = {key.rsplit("::", 1)[0] for key in ewc_fisher if key.endswith("::u")}
                if expected_sites != fisher_sites:
                    raise ValueError("EWC Fisher site paths do not exactly match this initialization's LoRA sites.")
                ewc_penalty = jnp.zeros((), jnp.float32)
                ewc_sites = 0
                for a_path, a in trainable.items():
                    if not a_path.endswith("lora_a") or f"{a_path}::u" not in ewc_fisher:
                        continue
                    b = trainable[a_path[:-1] + "b"]
                    u, v, mass = (ewc_fisher[f"{a_path}::{suffix}"] for suffix in ("u", "v", "mass"))
                    a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
                    gram_a = jnp.einsum("...ir,...is,...i->...rs", a32, a32, u)
                    gram_b = jnp.einsum("...rj,...sj,...j->...rs", b32, b32, v)
                    ewc_penalty = ewc_penalty + jnp.sum(jnp.sum(gram_a * gram_b, axis=(-2, -1)) / mass)
                    ewc_sites += 1
                if ewc_sites != 10:
                    raise ValueError(f"EWC Fisher matched {ewc_sites} LoRA sites; expected 10.")
                loss = loss + 0.5 * config.ewc_lambda * ewc_penalty
                aux["ewc_penalty"] = ewc_penalty.astype(jnp.float32)
                aux["ewc_sites"] = jnp.asarray(ewc_sites, jnp.float32)
            return loss.astype(jnp.float32), aux
'''


def patched_trainer_path(openpi_root: Path, source_hashes: dict[str, str]) -> Path:
    source = Path(openpi_root) / "scripts/train.py"
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source_hashes["scripts/train.py"]
    if actual != expected:
        raise ValueError(f"Pinned trainer hash mismatch: {actual} != {expected}")
    original = source.read_text()
    if original.count(CPD_RETURN) != 1:
        raise ValueError("Pinned trainer CPD return anchor changed; refusing an unsafe overlay patch")
    patched = original.replace(CPD_RETURN, CPD_CONTEXT + EWC_BLOCK, 1)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    overlay = Path(__file__).resolve().parent / "generated" / digest / "train.py"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if overlay.exists() and overlay.read_text() != patched:
        raise ValueError("Generated trainer overlay hash collision")
    if not overlay.exists():
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=overlay.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(patched)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, overlay)
        except FileExistsError:
            if overlay.read_text() != patched:
                raise ValueError("Concurrent generated trainer differs from requested overlay")
        finally:
            temporary.unlink(missing_ok=True)
    return overlay
