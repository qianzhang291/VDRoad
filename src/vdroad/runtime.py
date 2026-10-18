"""SAM-Road runtime integration and deterministic execution utilities."""
from __future__ import annotations

import contextlib
import importlib
import io
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def configure_determinism(seed: int) -> dict:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        if torch.cuda.is_initialized():
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"seed": seed, "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "matmul_tf32": False, "cudnn_tf32": False}


class _FixedGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grid):
        if grid.requires_grad:
            raise ValueError("Only fixed graph coordinates are supported")
        if not bool(torch.isfinite(grid).all()):
            raise ValueError("Graph coordinates must be finite")
        ctx.save_for_backward(grid)
        ctx.input_shape = features.shape
        ctx.input_dtype = features.dtype
        return F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.is_cuda and (not torch.are_deterministic_algorithms_enabled()
                                    or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError("CUDA backward requires strict deterministic algorithms")
        grid, = ctx.saved_tensors
        batch, channels, height, width = ctx.input_shape
        dtype = grad_output.dtype
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            coords = grid.to(dtype).reshape(batch, -1, 2)
            x = ((coords[..., 0] + 1) / 2) * width - 0.5
            y = ((coords[..., 1] + 1) / 2) * height - 0.5
            x0, y0 = x.floor(), y.floor()
            dx, dy = x - x0, y - y0
            grad = grad_output.to(dtype).reshape(batch, channels, -1)
            result = grad.new_zeros((batch, channels, height * width))
            for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                                   (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
                xx, yy = x0 + ox, y0 + oy
                valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                index = (yy.clamp(0, height-1).long() * width
                         + xx.clamp(0, width-1).long())
                contribution = grad * (weight * valid).unsqueeze(1)
                result.scatter_add_(2, index.unsqueeze(1).expand(-1, channels, -1), contribution)
            return result.reshape(ctx.input_shape).to(ctx.input_dtype), None


class FixedPointBilinearSampler(nn.Module):
    """Original SAM-Road forward with deterministic fixed-coordinate gradients."""
    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size

    def forward(self, feature_maps, sample_points):
        if sample_points.requires_grad:
            raise ValueError("Only fixed graph coordinates are supported")
        grid = ((sample_points / self.patch_size) * 2.0 - 1.0).unsqueeze(2)
        sampled = _FixedGridSample.apply(feature_maps, grid)
        return sampled.squeeze(-1).permute(0, 2, 1)


def head_kwargs(config: dict) -> dict:
    """Read the shared training/inference constructor parameters."""
    spec = config.get("completion", config)
    keys = ("feature_channels", "hidden_dim", "num_predictions", "roi_size",
            "roi_radius_pixels", "heads", "layers", "ffn_dim", "dropout")
    values = {key: spec[key] for key in keys if key in spec}
    values["patch_size"] = config.get("native_graph_protocol", {}).get(
        "PATCH_SIZE", spec.get("patch_size", 256)
    )
    return values


def endpoint_generator(dataset, config, raw_graph, transform):
    """Preserve the formal training recipe's empty-graph handling."""
    if raw_graph:
        return dataset.GraphLabelGenerator(config, raw_graph, transform)
    import igraph
    import rtree
    from types import SimpleNamespace
    return SimpleNamespace(full_graph_subdivide=igraph.Graph(),
                           subdivide_points=np.empty((0, 2), np.float32),
                           graph_rtee=rtree.index.Index())


def load_runtime(config: dict, batch: int = 96, samples_per_epoch: int = 24576) -> dict:
    """Import an explicitly supplied SAM-Road checkout and its native dataset API.

    Paths must already be absolute (the training CLI resolves them). The upstream
    modules use unqualified imports, so conflicting preloaded modules are rejected.
    """
    paths = config["paths"]
    required = ("external_samroad_root", "baseline_config", "sam_checkpoint")
    for key in required:
        if not paths.get(key):
            raise ValueError(f"Missing paths.{key}; supply the corresponding CLI path")
    external = Path(paths["external_samroad_root"]).expanduser().resolve()
    for name in ("dataset", "model", "utils", "graph_utils", "graph_extraction"):
        expected = external / f"{name}.py"
        if not expected.is_file():
            raise FileNotFoundError(expected)
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != expected:
            raise RuntimeError(f"Module {name!r} was imported from another checkout; use a fresh process")
    for key in ("baseline_config", "sam_checkpoint"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])
    sys.path.insert(0, str(external))
    os.environ.setdefault("WANDB_MODE", "disabled")
    dataset = importlib.import_module("dataset")
    model = importlib.import_module("model")
    utils = importlib.import_module("utils")
    sam_config = utils.load_config(str(paths["baseline_config"]))
    sam_config.SAM_CKPT_PATH = str(paths["sam_checkpoint"])
    sam_config.BATCH_SIZE = int(batch)
    if (sam_config.PATCH_SIZE != 256 or sam_config.DATASET != "spacenet"
            or sam_config.USE_SAM_DECODER or sam_config.ENCODER_LORA
            or sam_config.FOCAL_LOSS is not False):
        raise ValueError("Expected the native SpaceNet ViT-B/256 SAM-Road configuration")
    for key, expected in config["native_graph_protocol"].items():
        setattr(sam_config, key, expected)
    return {"external": external, "sam_config": sam_config, "SAMRoad": model.SAMRoad,
            "GraphLabelGenerator": lambda c, g, t: endpoint_generator(dataset, c, g, t),
            "get_patch_info_one_img": dataset.get_patch_info_one_img,
            "graph_collate_fn": dataset.graph_collate_fn, "read_rgb_img": dataset.read_rgb_img,
            "graph_utils": importlib.import_module("graph_utils"),
            "graph_extraction": importlib.import_module("graph_extraction"),
            "samples_per_epoch": int(samples_per_epoch)}


def load_frozen_base(runtime: dict, checkpoint: str | Path, seed: int = 20260916,
                     device: str | torch.device = "cuda:0") -> nn.Module:
    """Load a trusted local SAM-Road base checkpoint, then freeze every parameter.

    Checkpoints may store weights under ``model_state_dict`` or ``state_dict``;
    a plain state dictionary is also accepted.
    Loading is strict; there is no random-weight fallback.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    configure_determinism(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        base = runtime["SAMRoad"](runtime["sam_config"])
    base.bilinear_sampler = FixedPointBilinearSampler(runtime["sam_config"].PATCH_SIZE)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    base.load_state_dict(state, strict=True)
    base.requires_grad_(False)
    return base.to(device).eval()
