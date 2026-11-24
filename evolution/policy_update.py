"""Train one H2P candidate using native per-example PyTorch policy losses.

Invoke with --adapter package.module:factory and an optional --adapter-config
JSON. The factory receives that config dictionary. It returns an object with:

* load_policy(checkpoint): a torch.nn.Module on the intended device, with the
  trainable subset selected by requires_grad.
* supervised_batch(records, rng): resolve episode evidence, sample targets and
  prepare the model inputs. rng is a Python random.Random.
* supervised_losses(model, batch): return one differentiable native loss per
  supervised example after reduction over its valid target elements.
* preference_batch(records, rng): load both executed chunks at the SAME public
  observation, along with masks and shared random draws for stochastic losses.
* preference_losses(model, batch): return (positive_loss, negative_loss),
  each a vector of per-example losses. Both targets must use the same draws.
* save_policy(model, directory): publish the native weights and inference
  assets to the provided empty directory, loadable by load_policy/evaluate.

This driver receives the training-stage JSON emitted by evolution.run. The
adapter handles the policy-specific tensor format, action convention, native
loss and checkpoint serialization. The driver owns optimization and sampling.
"""
import argparse
import copy
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch

from .evidence import sample_batch


@dataclass(frozen=True)
class Options:
    steps: int
    batch_size: int
    pair_batch_size: int
    learning_rate: float
    fisher_samples: int
    lambda_ewc: float
    lambda_pref: float
    lambda_prox: float
    margin: float
    slack: float
    seed: int


def _per_example(losses, count):
    if not torch.is_tensor(losses) or losses.shape != (count,):
        raise ValueError("Native loss must be a tensor with one scalar per example")
    if not torch.isfinite(losses.detach()).all():
        raise ValueError("Nonfinite native loss")
    return losses.float()


def preference_objective(positive, negative, frozen_positive, frozen_negative, *, margin, slack):
    """Eq. 2 and 3 with a frozen round-start reference and mean over examples."""
    positive_change = positive - frozen_positive.detach()
    negative_change = negative - frozen_negative.detach()
    gap = (positive_change - negative_change + margin).relu().square().mean()
    prox = (positive_change.relu().square() + (negative_change - slack).relu().square()).mean()
    return gap, prox


def empirical_fisher(model, adapter, records, count, rng):
    """Mean of squared single-example gradients on the previous data bank."""
    if not records or count < 1:
        raise ValueError("EWC requires prior demonstrations and a positive Fisher sample count")
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    information = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in params.items()}
    model.eval()
    for _ in range(count):
        batch = adapter.supervised_batch([rng.choice(records)], rng)
        loss = _per_example(adapter.supervised_losses(model, batch), 1)[0]
        grads = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
        for (name, _), grad in zip(params.items(), grads):
            if grad is not None:
                information[name].add_(grad.detach().float().square())
    for value in information.values():
        value.div_(count)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite Fisher importance")
    return information


def ewc_penalty(parameters, anchors, information):
    """Fisher-weighted squared displacement, before the lambda/2 multiplier."""
    return sum((importance * (parameters[key].float() - anchors[key].float()).square()).sum()
               for key, importance in information.items())


def _paired_losses(adapter, model, reference, batch, count, rng):
    seed = rng.getrandbits(62)
    cuda_devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})

    def forward(network):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            for device in cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            positive, negative = adapter.preference_losses(network, batch)
            return _per_example(positive, count), _per_example(negative, count)

    model.eval()
    try:
        positive, negative = forward(model)
        with torch.no_grad():
            frozen_positive, frozen_negative = forward(reference)
    finally:
        model.train()
    return positive, negative, frozen_positive, frozen_negative


def update(request, adapter, options, output_dir):
    if (request["policy"] != request["reference"] or
            type(request["cpd_enabled"]) is not bool or type(request["ewc"]) is not bool):
        raise ValueError("Training request must bind the round-start reference and explicit loss switches")
    pool = request["pool"]
    probabilities = request["task_probabilities"]
    if not pool["demonstrations"] or request["cpd_enabled"] != bool(pool["pairs"]):
        raise ValueError("SFT requires admitted demonstrations; CPD follows compatible pairs")
    if options.steps < 1 or options.batch_size < 1 or (pool["pairs"] and options.pair_batch_size < 1):
        raise ValueError("Training steps and active batch sizes must be positive")
    for weight in (options.lambda_ewc, options.lambda_pref, options.lambda_prox, options.margin, options.slack):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Loss weights, margin and slack must be finite and nonnegative")
    if not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    seed = options.seed + request["round"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    policy = adapter.load_policy(request["policy"])
    parameters = dict(policy.named_parameters())
    trainable = [p for p in parameters.values() if p.requires_grad]
    if not trainable:
        raise ValueError("The policy adapter exposed no trainable parameters")
    ref = copy.deepcopy(policy).eval()
    ref.requires_grad_(False)
    anchors = dict(ref.named_parameters())
    fisher = {}
    if request["ewc"] and options.lambda_ewc:
        fisher = empirical_fisher(policy, adapter, request["fisher_demonstrations"],
                                  options.fisher_samples, random.Random(seed))
    optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=0.)
    policy.train()
    for _ in range(options.steps):
        examples = sample_batch(pool["demonstrations"], probabilities["demonstrations"], options.batch_size, rng)
        batch = adapter.supervised_batch(examples, rng)
        loss = _per_example(adapter.supervised_losses(policy, batch), len(examples)).mean()
        if pool["pairs"] and options.lambda_pref:
            pairs = sample_batch(pool["pairs"], probabilities["pairs"], options.pair_batch_size, rng)
            pair_batch = adapter.preference_batch(pairs, rng)
            losses = _paired_losses(adapter, policy, ref, pair_batch, len(pairs), rng)
            gap, prox = preference_objective(*losses, margin=options.margin, slack=options.slack)
            loss = loss + options.lambda_pref * (gap + options.lambda_prox * prox)
        if fisher:
            loss = loss + options.lambda_ewc / 2 * ewc_penalty(parameters, anchors, fisher)
        if not torch.isfinite(loss.detach()):
            raise ValueError("Nonfinite total loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None):
            raise ValueError("Nonfinite policy gradient")
        optimizer.step()

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as scratch:
        staging = Path(scratch) / "checkpoint"
        staging.mkdir()
        adapter.save_policy(policy.eval(), staging)
        if not any(p.is_file() and p.stat().st_size for p in staging.rglob("*")):
            raise ValueError("Adapter did not publish checkpoint files")
        staging.rename(output_dir)
    return {"policy": str(output_dir), "reference": request["reference"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="module:factory for native policy losses and I/O")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--fisher-samples", type=int, default=500)
    parser.add_argument("--lambda-ewc", type=float, default=4e-6)
    parser.add_argument("--lambda-pref", type=float, default=0.02)
    parser.add_argument("--lambda-prox", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--slack", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    module_name, factory_name = args.adapter.split(":", 1)
    config = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
    adapter = getattr(importlib.import_module(module_name), factory_name)(config)
    options = Options(**{k: getattr(args, k) for k in Options.__dataclass_fields__})
    request = json.loads(args.request.read_text())
    report = update(request, adapter, options, Path(request["workdir"]) / "candidate_policy")
    args.response.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
