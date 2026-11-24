#!/usr/bin/env python3
"""Run the OpenPI LIBERO v8 EWC recipes with local inputs."""

import argparse
import dataclasses
import importlib
import logging
import math
import os
from pathlib import Path, PurePosixPath
import sys


DENSE = "pi05_libero_v8_paper_ewc_full"
CONTROL = "pi05_libero_v8_paper_ewc_full_control"
LORA = ("pi05_libero_v8_ewc_lora", "pi05_libero_v8_ewc_r256_lora")
TASK_A_REPO = "physical-intelligence/libero"


def local_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train or resume an EWC recipe")
    fisher = commands.add_parser("fisher", help="estimate Fisher weights on task-A data")
    train.add_argument("recipe", choices=(*LORA, DENSE, CONTROL))
    fisher.add_argument("recipe", choices=(*LORA, DENSE))

    for command in (train, fisher):
        command.add_argument("--openpi-root", type=local_path, required=True)
        command.add_argument("--data-root", type=local_path, required=True, help="LeRobot v2.1 data home")
        command.add_argument("--repo-id", help="dataset ID under the data root")
        command.add_argument("--assets", type=local_path, required=True, help="normalization assets directory")
        command.add_argument("--params", type=local_path, required=True, help="task-A Orbax params directory")
        command.add_argument("--fsdp-devices", type=positive_int)
        command.add_argument("--check-only", action="store_true", help="validate local inputs without loading weights")

    train.add_argument("--output-root", type=local_path, required=True)
    train.add_argument("--exp-name", required=True)
    train.add_argument("--episode-filter", type=local_path, help="required task-B episode JSON for the dense recipes")
    train.add_argument("--fisher", type=local_path, help="Orbax params directory for dense EWC, or NPZ file for LoRA")
    train.add_argument("--ewc-lambda", type=float, help="override the original config's coefficient")
    train.add_argument("--steps", type=positive_int, help="override the total optimizer step count")
    train.add_argument("--resume", action="store_true")
    fisher.add_argument("--out", type=local_path, required=True, help="new Fisher params directory or LoRA NPZ file")
    fisher.add_argument("--steps", type=positive_int, help="default 512 dense examples or 500 LoRA batches")
    fisher.add_argument("--batch-size", type=positive_int, help="LoRA Fisher batch size, default 4")
    return root


def check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_relative_id(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"expected a relative dataset ID, got {value!r}")
    return value


def load_config(args):
    for path in (args.openpi_root / "pyproject.toml", args.params / "_METADATA"):
        check_file(path)
    os.environ["HF_LEROBOT_HOME"] = str(args.data_root)
    sys.path[:0] = [
        str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
        str(Path(__file__).resolve().parent.parent / "libero_v8"),
    ]
    if not args.check_only:
        importlib.import_module("openpi_train")
    from openpi.shared import normalize
    from openpi.training import config, weight_loaders

    source = config.get_config(args.recipe)
    if source.cpd_enabled or source.lwf_enabled or source.vra_enabled:
        raise ValueError("expected a behavior-cloning EWC recipe")
    is_training = args.command == "train"
    dense = args.recipe in (DENSE, CONTROL)
    repo_id = require_relative_id(args.repo_id or (source.data.repo_id if is_training else TASK_A_REPO))
    check_file(args.data_root / repo_id / "meta/info.json")
    asset_id = source.data.assets.asset_id
    normalize.load(args.assets / asset_id)
    episode_filter = args.episode_filter if is_training else None
    if is_training and dense and episode_filter is None:
        raise ValueError("dense task-B training requires --episode-filter")
    if episode_filter is not None:
        check_file(episode_filter)

    data = dataclasses.replace(
        source.data,
        repo_id=repo_id,
        assets=dataclasses.replace(source.data.assets, assets_dir=str(args.assets)),
        base_config=dataclasses.replace(
            source.data.base_config,
            lerobot_episode_filter_path=str(episode_filter) if episode_filter else None,
        ),
    )
    cfg = dataclasses.replace(
        source,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.params)),
        fsdp_devices=args.fsdp_devices or source.fsdp_devices,
        ewc_fisher_path=None,
        wandb_enabled=False,
    )

    if not is_training:
        if args.out.exists():
            raise FileExistsError(args.out)
        if dense:
            if args.batch_size not in (None, 1):
                raise ValueError("dense Fisher requires batch size one")
            if args.out.name != "params":
                raise ValueError("use a new <fisher-directory>/params directory for dense Fisher")
            if (args.out.parent / "manifest.json").exists():
                raise FileExistsError(args.out.parent / "manifest.json")
            return dataclasses.replace(cfg, batch_size=1, num_workers=0)
        if args.out.suffix != ".npz":
            raise ValueError("LoRA Fisher output must use the .npz extension")
        return dataclasses.replace(cfg, batch_size=args.batch_size or 4)

    if Path(args.exp_name).name != args.exp_name or args.exp_name in {"", ".", ".."}:
        raise ValueError("--exp-name must be a single directory name")
    if cfg.ewc_enabled:
        if args.fisher is None:
            raise ValueError("EWC training requires --fisher")
        check_file(args.fisher / "_METADATA" if dense else args.fisher)
    elif args.fisher is not None or args.ewc_lambda is not None:
        raise ValueError("the matched BC control accepts neither --fisher nor --ewc-lambda")
    if args.ewc_lambda is not None and (not math.isfinite(args.ewc_lambda) or args.ewc_lambda < 0):
        raise ValueError("--ewc-lambda must be finite and nonnegative")
    return dataclasses.replace(
        cfg,
        checkpoint_base_dir=str(args.output_root),
        exp_name=args.exp_name,
        num_train_steps=args.steps or cfg.num_train_steps,
        ewc_fisher_path=str(args.fisher) if args.fisher else None,
        ewc_lambda=cfg.ewc_lambda if args.ewc_lambda is None else args.ewc_lambda,
        resume=args.resume,
        overwrite=False,
    )


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        cfg = load_config(args)
    except (ValueError, OSError) as error:
        cli.error(str(error))

    if args.check_only:
        print(f"{cfg.name}: {args.command} inputs validated, dataset={cfg.data.repo_id}")
        return

    from lerobot_datasets_compat import install

    install()
    logging.basicConfig(level=logging.INFO)
    if args.command == "train":
        import openpi_train

        openpi_train.main(cfg)
    elif cfg.ewc_full_diagonal:
        from compute_paper_ewc_full_fisher import compute

        compute(cfg, args.out, args.steps or 512, seed=20260818)
    else:
        from compute_ewc_fisher import compute

        compute(cfg, args.out, args.steps or 500)


if __name__ == "__main__":
    main()
