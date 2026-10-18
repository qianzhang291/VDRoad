"""Portable FP32 inference using cached SAM-Road patch features."""
from __future__ import annotations

import numpy as np
import torch

from .core import EndpointCompletionHead
from .graph import expand_graph
from .runtime import configure_determinism, head_kwargs


def best_patch_indices(points_xy, origins_xy, patch_size=256):
    """Largest boundary clearance, with the earliest patch breaking ties."""
    result = []
    for x, y in np.asarray(points_xy):
        choices = []
        for index, (x0, y0) in enumerate(np.asarray(origins_xy)):
            x1, y1 = x0 + patch_size, y0 + patch_size
            if x0 <= x <= x1 - 1 and y0 <= y <= y1 - 1:
                margin = min(x - x0, x1 - 1 - x, y - y0, y1 - 1 - y)
                choices.append((float(margin), -index, index))
        if not choices:
            raise ValueError(f"Endpoint {(float(x), float(y))} has no covering feature patch")
        result.append(max(choices)[2])
    return np.asarray(result, dtype=np.int64)


def load_head(checkpoint, device="cpu", mode="full", config=None):
    """Load a trusted selected-head checkpoint or a plain tensor state dict."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "mode" in payload and payload["mode"] != mode:
        raise ValueError("Checkpoint mode does not match the requested ablation")
    state = payload.get("head_state_dict", payload)
    saved_config = payload.get("head_config") or payload.get("identity", {}).get("config")
    kwargs = head_kwargs(config or saved_config or {})
    if config is not None and saved_config is not None:
        if head_kwargs(saved_config) != kwargs:
            raise ValueError("Checkpoint architecture differs from the supplied configuration")
    seed = int(payload.get("seed", payload.get("identity", {}).get("seed", 20260916)))
    configure_determinism(seed + 700000)
    head = EndpointCompletionHead(**kwargs)
    head.load_state_dict(state, strict=True)
    return head.to(device=device, dtype=torch.float32).eval()


@torch.inference_mode()
def complete_graph(nodes_xy, edges, features, origins_xy, image_size_hw, head,
                   *, mode="full", rounds=3, score_threshold=0.5,
                   merge_distance=10.0, same_direction_cosine=0.94,
                   max_step_pixels=30.0):
    """Complete an in-image graph without consulting labels or metrics.

    Feature patches must be 256-pixel crops in canonical SAM-Road order.
    Coordinates use image pixels (x right, y down); edges index nodes_xy.
    """
    device = next(head.parameters()).device
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    origins = np.asarray(origins_xy, dtype=np.float32)
    height, width = map(int, image_size_hw)
    if features.ndim != 4 or features.shape[1] != 256:
        raise ValueError("Features must have shape [P,256,Hf,Wf]")
    if origins.shape != (len(features), 2) or not len(origins):
        raise ValueError("origins_xy must have shape [P,2] for nonempty features")
    if height < 1 or width < 1 or not np.isfinite(origins).all():
        raise ValueError("Invalid image size or patch coordinates")
    if not torch.isfinite(features).all():
        raise ValueError("Feature cache contains nonfinite values")
    nodes = np.asarray(nodes_xy, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError("nodes_xy must be finite [N,2] pixel coordinates")
    if np.any(nodes < 0) or np.any(nodes[:, 0] >= width) or np.any(nodes[:, 1] >= height):
        raise ValueError("This CLI expects in-image input vertices")
    head.eval()

    def predict(_ids, points_xy, incoming, _round):
        indices = best_patch_indices(points_xy, origins)
        local = points_xy.astype(np.float32) - origins[indices]
        selected = features[torch.as_tensor(indices, device=device)]
        query = torch.as_tensor(local[:, None, :], dtype=torch.float32, device=device)
        direction = torch.as_tensor(incoming[:, None, :], dtype=torch.float32, device=device)
        valid = torch.ones((len(points_xy), 1), dtype=torch.bool, device=device)
        logits, normalized = head(selected, query, direction, valid, mode=mode)
        probability = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
        delta = normalized[:, 0].float().cpu().numpy() * float(max_step_pixels)
        if not np.isfinite(probability).all() or not np.isfinite(delta).all():
            raise ValueError("Nonfinite completion prediction")
        return probability, delta

    return expand_graph(
        nodes, edges, predict, rounds=rounds, score_threshold=score_threshold,
        merge_distance=merge_distance, same_direction_cosine=same_direction_cosine,
        mode="learned_dynamic", bounds_xy=(width, height),
    )
