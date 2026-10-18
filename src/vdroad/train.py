"""Train completion heads (`full`, `visual_only`, `direction_only`) over a frozen SAM-Road base."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import EndpointCompletionHead, permutation_invariant_set_loss
from .data import EndpointFitCorpus, make_schedule, schedule_sha256
from .runtime import configure_determinism, head_kwargs, load_frozen_base, load_runtime


MODES = ("full", "visual_only", "direction_only")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, value, *, tensor: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if tensor:
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_split(path: str | Path, role: str) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value = value.get("roles", value)[role]
        result = value
    else:
        result = [line.strip() for line in path.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(result, list) or not result or not all(isinstance(x, str) for x in result):
        raise ValueError(f"Invalid {role} scene list: {path}")
    if len(result) != len(set(result)) or any("/" in x or "\\" in x for x in result):
        raise ValueError(f"Duplicate or invalid scene identifier in {path}")
    return result


def build_heads(config: dict, seed: int, device: torch.device) -> dict:
    modes = tuple(config["completion"].get("modes", ["full"]))
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError(f"Invalid completion modes: {modes}")
    def construct():
        configure_determinism(seed + 700000)
        return EndpointCompletionHead(**head_kwargs(config))
    initial = copy.deepcopy(construct().state_dict())
    heads = {mode: construct().to(device) for mode in modes}
    for head in heads.values():
        head.load_state_dict(initial, strict=True)
    return heads


def build_schedule(*, seed: int, role: str, scene_count: int,
                   samples_per_epoch: int, epochs: int, fixed_holdout: bool,
                   provided: Path | None = None) -> dict:
    """Use the experiment's complete train schedule and repeated inner patches."""
    if provided:
        with np.load(provided, allow_pickle=False) as archive:
            schedule = {key: archive[key].copy() for key in archive.files}
    elif role == "holdout" and fixed_holdout:
        first = make_schedule(seed, scene_count, samples_per_epoch)
        first["rotation"][:] = 0
        schedule = {key: np.tile(value, epochs) for key, value in first.items()}
    else:
        schedule = make_schedule(seed, scene_count, samples_per_epoch * epochs)
    if set(schedule) != {"scene_index", "x0", "y0", "rotation", "graph_seed"}:
        raise ValueError(f"Invalid {role} schedule keys")
    if any(value.ndim != 1 or len(value) != samples_per_epoch * epochs
           for value in schedule.values()):
        raise ValueError(f"Invalid {role} schedule length")
    if np.any(schedule["scene_index"] < 0) or np.any(schedule["scene_index"] >= scene_count):
        raise ValueError(f"Out-of-range {role} schedule scene index")
    if role == "holdout" and fixed_holdout:
        if np.any(schedule["rotation"] != 0) or any(
            not np.array_equal(value, np.tile(value[:samples_per_epoch], epochs))
            for value in schedule.values()
        ):
            raise ValueError("Holdout schedule must repeat fixed, unrotated patches")
    return schedule


def autocast_context(device: torch.device, precision: str):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if precision == "bf16" else contextlib.nullcontext())


def batch_loss(head, features, batch: dict, mode: str, config: dict,
               device: torch.device, precision: str) -> dict:
    spec = config["completion"]
    moved = {key: batch[key].to(device, non_blocking=True) for key in
             ("query_points", "incoming_directions", "target_deltas", "target_valid", "query_valid")}
    with autocast_context(device, precision):
        logits, deltas = head(features, moved["query_points"], moved["incoming_directions"],
                             moved["query_valid"], mode=mode)
        losses = permutation_invariant_set_loss(
            logits.float(), deltas.float(), moved["target_valid"],
            moved["target_deltas"].float(), moved["query_valid"],
            class_weight=float(spec["class_loss_weight"]),
            coordinate_weight=float(spec["coordinate_loss_weight"]),
            direction_weight=float(spec["direction_loss_weight"]),
        )
    if not bool(torch.isfinite(losses["total"]).item()):
        raise FloatingPointError(f"Nonfinite {mode} loss")
    return losses


def run_epoch(*, base, heads: dict, corpus, batch_size: int, workers: int, seed: int,
              epoch: int, config: dict, device: torch.device, precision: str,
              optimizers: dict | None) -> dict:
    training = optimizers is not None
    modes = tuple(heads)
    corpus.set_epoch(epoch)
    base.eval()
    for head in heads.values():
        head.train(training)
    kwargs = {"prefetch_factor": 2} if workers else {}
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=False,
                        generator=torch.Generator().manual_seed(seed + epoch + (0 if training else 900000)),
                        **kwargs)
    sums = {mode: {key: 0.0 for key in ("total", "class", "coord", "direction")}
            | {"samples": 0, "queries": 0, "targets": 0} for mode in modes}
    started = time.perf_counter()
    for item in loader:
        with autocast_context(device, precision):
            _, features = base.infer_masks_and_img_features(item["rgb"].to(device, non_blocking=True))
            features = features.detach()
        for mode in modes:
            if training:
                optimizers[mode].zero_grad(set_to_none=True)
                losses = batch_loss(heads[mode], features, item, mode, config, device, precision)
                losses["total"].backward()
                gradients = [p.grad for p in heads[mode].parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError(f"Missing or nonfinite {mode} gradients")
                optimizers[mode].step()
            else:
                with torch.inference_mode():
                    losses = batch_loss(heads[mode], features, item, mode, config, device, precision)
            size = int(item["rgb"].shape[0])
            for key in ("total", "class", "coord", "direction"):
                sums[mode][key] += float(losses[key].detach().item()) * size
            sums[mode]["samples"] += size
            sums[mode]["queries"] += int(item["query_valid"].sum().item())
            sums[mode]["targets"] += int(item["target_valid"].sum().item())
    elapsed = time.perf_counter() - started
    return {mode: {key: value / sums[mode]["samples"]
                   if key in ("total", "class", "coord", "direction") else value
                   for key, value in sums[mode].items()} | {"seconds": elapsed}
            for mode in modes}


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/default.json"),
                        help="Config path relative to the caller's cwd (default: configs/default.json)")
    for flag in ("samroad-root", "samroad-config", "sam-checkpoint", "data-root"):
        result.add_argument("--" + flag, type=Path)
    result.add_argument("--base-checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--train-split", type=Path, help="Ordered txt IDs or JSON roles.train; defaults to inner_manifest")
    result.add_argument("--holdout-split", type=Path, help="Ordered txt IDs or JSON roles.holdout; defaults to inner_manifest")
    result.add_argument("--train-schedule", type=Path, help="Optional precomputed head_train NPZ schedule")
    result.add_argument("--holdout-schedule", type=Path, help="Optional precomputed head_holdout NPZ schedule")
    result.add_argument("--seed", type=int, default=20260916)
    result.add_argument("--modes", nargs="+", choices=MODES,
                        help="Heads to train; defaults to completion.modes (full)")
    result.add_argument("--batch-size", type=int, default=96)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--resume", action="store_true", help="Resume last.pt at its completed epoch boundary")
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    if args.modes is not None:
        config["completion"]["modes"] = args.modes
    modes = tuple(config["completion"].get("modes", ["full"]))
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError(f"Invalid completion modes: {modes}")
    # A configs/ directory belongs to its parent release root. Other config files
    # resolve their paths from their own directory; CLI paths use the caller's cwd.
    release_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for key, value in config["paths"].items():
        if value:
            path = Path(value).expanduser()
            config["paths"][key] = str((path if path.is_absolute() else release_root / path).resolve())
    for flag, key in (("samroad_root", "external_samroad_root"), ("samroad_config", "baseline_config"),
                      ("sam_checkpoint", "sam_checkpoint"), ("data_root", "data_root")):
        if getattr(args, flag) is not None:
            config["paths"][key] = str(getattr(args, flag).expanduser().resolve())
    for key in ("external_samroad_root", "baseline_config", "sam_checkpoint", "data_root"):
        if not config["paths"].get(key):
            raise ValueError(f"Missing paths.{key}; supply its CLI path or set it in the config")
    device = torch.device(args.device)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive; workers must be nonnegative")
    configure_determinism(args.seed)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training requires a BF16-capable CUDA device; use --precision fp32 on CPU")
    paths, spec = config["paths"], config["completion"]
    train_ids = read_split(args.train_split or paths["inner_manifest"], "train")
    holdout_ids = read_split(args.holdout_split or paths["inner_manifest"], "holdout")
    permitted = set(read_split(paths["prospective_manifest"], "fit"))
    if set(train_ids) & set(holdout_ids) or not set(train_ids + holdout_ids) <= permitted:
        raise ValueError("Train/holdout must be disjoint subsets of the explicit FIT manifest")
    epochs = int(spec["max_epochs"])
    if epochs < 1:
        raise ValueError("completion.max_epochs must be positive")
    schedules = {}
    for role, ids, offset, count in (("train", train_ids, 200000, int(spec["samples_per_epoch"])),
                                    ("holdout", holdout_ids, 300000, int(spec["holdout_samples"]))):
        schedules[role] = build_schedule(
            seed=args.seed + offset, role=role, scene_count=len(ids),
            samples_per_epoch=count, epochs=epochs,
            fixed_holdout=bool(spec.get("fixed_holdout", True)),
            provided=getattr(args, role + "_schedule"),
        )
    output = args.output_dir.expanduser().resolve()
    checkpoint = args.base_checkpoint.expanduser().resolve()
    learning_rate = float(spec["learning_rate"]) * args.batch_size / 64.0
    identity = {"config": config, "seed": args.seed, "batch_size": args.batch_size,
                "workers": args.workers, "device": str(device), "precision": args.precision,
                "learning_rate": learning_rate, "mode_order": list(modes),
                "base_checkpoint": str(checkpoint), "base_checkpoint_sha256": sha256_file(checkpoint),
                "train_ids": train_ids, "holdout_ids": holdout_ids,
                "schedules": {role: schedule_sha256(value) for role, value in schedules.items()},
                "source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ("core.py", "data.py", "runtime.py", "train.py")},
                "samroad_inputs_sha256": {key: sha256_file(paths[key])
                                          for key in ("baseline_config", "sam_checkpoint")}}
    if args.resume:
        recovery = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if recovery["identity"] != identity:
            raise ValueError("Resume configuration, source, checkpoint, or schedule differs")
    else:
        recovery = None
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}; use --resume for a matching last.pt")
        output.mkdir(parents=True, exist_ok=True)
    atomic_save(output / "config.json", identity)
    for role, value in schedules.items():
        if not args.resume:
            np.savez_compressed(output / f"head_{role}.npz", **value)
    runtime = load_runtime(config, args.batch_size, int(spec["samples_per_epoch"]))
    corpora = {role: EndpointFitCorpus(config=config, scene_ids=ids, schedule=schedules[role],
                                      runtime=runtime, samples_per_epoch=count, max_epochs=epochs)
               for role, ids, count in (("train", train_ids, int(spec["samples_per_epoch"])),
                                        ("holdout", holdout_ids, int(spec["holdout_samples"])))}
    base = load_frozen_base(runtime, checkpoint, args.seed, device)
    heads = build_heads(config, args.seed, device)
    optimizers = {mode: torch.optim.AdamW(head.parameters(), lr=learning_rate,
                                         weight_decay=float(spec["weight_decay"]))
                  for mode, head in heads.items()}
    best = {mode: {"loss": math.inf, "epoch": 0, "stale": 0} for mode in modes}
    best_payloads, history, completed = {}, [], 0
    if recovery is not None:
        for mode in modes:
            heads[mode].load_state_dict(recovery["head_state_dicts"][mode], strict=True)
            optimizers[mode].load_state_dict(recovery["optimizer_state_dicts"][mode])
        best, best_payloads = recovery["best"], recovery["best_payloads"]
        history, completed = recovery["history"], int(recovery["completed_epoch"])
        restore_rng(recovery["rng_state"])
    for epoch in range(completed, epochs):
        # Check shared early-stopping patience across modes.
        if epoch >= int(spec["min_epochs"]) and all(best[m]["stale"] >= int(spec["patience"]) for m in modes):
            break
        rows = {}
        for role in ("train", "holdout"):
            rows[role] = run_epoch(base=base, heads=heads, corpus=corpora[role], batch_size=args.batch_size,
                                  workers=args.workers, seed=args.seed, epoch=epoch, config=config,
                                  device=device, precision=args.precision,
                                  optimizers=optimizers if role == "train" else None)
        for mode in modes:
            value = rows["holdout"][mode]["total"]
            if value < best[mode]["loss"] * (1 - float(spec["minimum_relative_improvement"])):
                best[mode] = {"loss": value, "epoch": epoch + 1, "stale": 0}
                best_payloads[mode] = {"seed": args.seed, "mode": mode, "epoch": epoch + 1,
                                      "head_config": head_kwargs(config),
                                      "head_state_dict": {k: v.detach().cpu().clone()
                                                          for k, v in heads[mode].state_dict().items()},
                                      "holdout": rows["holdout"][mode], "base_checkpoint": str(checkpoint)}
            else:
                best[mode]["stale"] += 1
        completed = epoch + 1
        history.append({"epoch": completed, **rows, "best": copy.deepcopy(best)})
        atomic_save(output / "last.pt", {"identity": identity, "completed_epoch": completed,
                    "head_state_dicts": {mode: head.state_dict() for mode, head in heads.items()},
                    "optimizer_state_dicts": {mode: opt.state_dict() for mode, opt in optimizers.items()},
                    "best": best, "best_payloads": best_payloads, "history": history,
                    "rng_state": capture_rng()}, tensor=True)
        atomic_save(output / "curves.json", history)
        for mode, payload in best_payloads.items():
            atomic_save(output / f"best_{mode}.pt", payload, tensor=True)
        print(f"epoch={completed} " + " ".join(f"{m}={rows['holdout'][m]['total']:.6f}" for m in modes), flush=True)
    # Derive exports again from the saved selection state if resuming after a write interruption.
    atomic_save(output / "curves.json", history)
    for mode, payload in best_payloads.items():
        atomic_save(output / f"best_{mode}.pt", payload, tensor=True)
    atomic_save(output / "summary.json", {"status": "COMPLETE", "seed": args.seed,
                "completed_epochs": completed, "mode_order": list(modes), "best": best,
                "batch_size": args.batch_size, "learning_rate": learning_rate, "precision": args.precision,
                "parameter_count_each": sum(p.numel() for p in next(iter(heads.values())).parameters()),
                "train_scene_count": len(train_ids), "holdout_scene_count": len(holdout_ids),
                "selection": "minimum inner-holdout set loss; earlier ties; schedule and epochs from config"})


if __name__ == "__main__":
    main()
