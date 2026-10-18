"""Core predecessor-conditioned endpoint completion module and set loss.

Consumes `[B, 256, H, W]` patch feature maps and predicts candidate continuation
branches in `(x, y)` pixel coordinates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


AblationMode = Literal["full", "visual_only", "direction_only"]
InferenceMode = Literal["learned_dynamic", "learned_snap"]


def _pair(value: int | float | Sequence[int | float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if len(value) != 2:
        raise ValueError("patch_size must be a scalar or (height, width)")
    return float(value[0]), float(value[1])


def sample_endpoint_rois(
    feature_map: Tensor,
    endpoint_xy: Tensor,
    *,
    patch_size: int | Sequence[int],
    roi_size: int = 5,
    roi_radius_pixels: float = 32.0,
) -> Tensor:
    """Bilinearly sample square endpoint ROIs from cached SAM features.

    Args:
        feature_map: ``[B,C,Hf,Wf]`` tensor.
        endpoint_xy: ``[B,N,2]`` patch-pixel coordinates in ``(x,y)`` order.
        patch_size: input patch ``(height,width)`` or a scalar for square input.
        roi_size: odd number of samples on each ROI side.
        roi_radius_pixels: distance from the center to an ROI side in patch
            pixels.  Samples outside the patch use zero padding.

    Returns:
        Tensor of shape ``[B,N,C,roi_size,roi_size]``.

    ``align_corners=True`` is deliberate: pixel coordinate 0 and coordinate
    ``size-1`` map exactly to feature-map boundaries, which makes the coordinate
    convention explicit and keeps 90-degree rotation behavior well defined.
    """

    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B,C,H,W]")
    if endpoint_xy.ndim != 3 or endpoint_xy.shape[-1] != 2:
        raise ValueError("endpoint_xy must have shape [B,N,2]")
    if endpoint_xy.shape[0] != feature_map.shape[0]:
        raise ValueError("feature_map and endpoint_xy batch sizes differ")
    if roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("roi_size must be a positive odd integer")
    patch_h, patch_w = _pair(patch_size)
    if patch_h <= 1 or patch_w <= 1:
        raise ValueError("patch height and width must exceed one pixel")

    batch, channels, _height, _width = feature_map.shape
    endpoint_count = endpoint_xy.shape[1]
    if endpoint_count == 0:
        return feature_map.new_empty(
            (batch, 0, channels, roi_size, roi_size)
        )
    offsets = torch.linspace(
        -float(roi_radius_pixels),
        float(roi_radius_pixels),
        roi_size,
        dtype=endpoint_xy.dtype,
        device=endpoint_xy.device,
    )
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    grid_x = endpoint_xy[..., 0, None, None] + offset_x
    grid_y = endpoint_xy[..., 1, None, None] + offset_y
    normalized_grid = torch.stack(
        (2.0 * grid_x / (patch_w - 1.0) - 1.0,
         2.0 * grid_y / (patch_h - 1.0) - 1.0),
        dim=-1,
    )

    sampled = F.grid_sample(
        feature_map,
        normalized_grid.reshape(batch, endpoint_count * roi_size, roi_size, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(
        batch, channels, endpoint_count, roi_size, roi_size
    ).permute(0, 2, 1, 3, 4)


class EndpointCompletionHead(nn.Module):
    """Predict an unordered set of up to three outgoing endpoint deltas.

    All ablations execute the exact same parameterized graph.  They differ only
    by zeroing either the sampled visual values or the incoming direction at the
    input boundary, so parameter counts and capacity are identical.
    """

    MODES = frozenset(("full", "visual_only", "direction_only"))

    def __init__(
        self,
        *,
        feature_channels: int = 256,
        hidden_dim: int = 128,
        num_predictions: int = 3,
        roi_size: int = 5,
        roi_radius_pixels: float = 32.0,
        patch_size: int | Sequence[int] = 256,
        heads: int = 4,
        layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_predictions != 3:
            raise ValueError("EndpointCompletionHead requires num_predictions=3")
        if roi_size <= 0 or roi_size % 2 == 0:
            raise ValueError("roi_size must be a positive odd integer")
        self.feature_channels = int(feature_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_predictions = int(num_predictions)
        self.roi_size = int(roi_size)
        self.roi_radius_pixels = float(roi_radius_pixels)
        self.patch_size = patch_size

        self.visual_projection = nn.Linear(feature_channels, hidden_dim)
        self.roi_position = nn.Parameter(
            torch.empty(roi_size * roi_size, hidden_dim)
        )
        self.direction_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_embedding = nn.Embedding(num_predictions, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 2)
        nn.init.trunc_normal_(self.roi_position, std=0.02)

    def forward(
        self,
        features: Tensor,
        query_points: Tensor,
        incoming_directions: Tensor,
        query_valid: Tensor,
        mode: AblationMode = "full",
    ) -> tuple[Tensor, Tensor]:
        """Return validity logits ``[B,N,3]`` and deltas ``[B,N,3,2]``.

        Deltas are component-wise normalized to ``[-1,1]``.  The inference
        adapter is responsible for multiplying them by the frozen maximum step
        in patch pixels.
        """

        if mode not in self.MODES:
            raise ValueError(f"unsupported ablation mode: {mode}")
        if features.ndim != 4 or features.shape[1] != self.feature_channels:
            raise ValueError(
                f"feature_map must have shape [B,{self.feature_channels},H,W]"
            )
        if query_points.ndim != 3 or query_points.shape[-1] != 2:
            raise ValueError("endpoint_xy must have shape [B,N,2]")
        if incoming_directions.shape != query_points.shape:
            raise ValueError("incoming_direction must match endpoint_xy")
        if query_valid.shape != query_points.shape[:2]:
            raise ValueError("endpoint_valid must have shape [B,N]")
        if query_points.shape[1] == 0:
            batch = features.shape[0]
            return (
                features.new_empty((batch, 0, self.num_predictions)),
                features.new_empty((batch, 0, self.num_predictions, 2)),
            )

        rois = sample_endpoint_rois(
            features,
            query_points,
            patch_size=self.patch_size,
            roi_size=self.roi_size,
            roi_radius_pixels=self.roi_radius_pixels,
        )
        directions = F.normalize(incoming_directions, dim=-1, eps=1e-6)
        if mode == "direction_only":
            rois = torch.zeros_like(rois)
        if mode == "visual_only":
            directions = torch.zeros_like(directions)

        batch, endpoints, channels, roi_h, roi_w = rois.shape
        memory = rois.permute(0, 1, 3, 4, 2).reshape(
            batch * endpoints, roi_h * roi_w, channels
        )
        memory = self.visual_projection(memory) + self.roi_position[None]
        direction_token = self.direction_projection(
            directions.reshape(batch * endpoints, 2)
        )
        queries = self.query_embedding.weight[None].expand(
            batch * endpoints, -1, -1
        )
        queries = queries + direction_token[:, None, :]
        decoded = self.output_norm(self.decoder(queries, memory))

        logits = self.validity_head(decoded).squeeze(-1).reshape(
            batch, endpoints, self.num_predictions
        )
        normalized_delta = torch.tanh(self.delta_head(decoded)).reshape(
            batch, endpoints, self.num_predictions, 2
        )
        valid = query_valid.to(dtype=torch.bool)
        logits = torch.where(valid[..., None], logits, torch.zeros_like(logits))
        normalized_delta = torch.where(
            valid[..., None, None], normalized_delta, torch.zeros_like(normalized_delta)
        )
        return logits, normalized_delta


def _safe_direction_cost(prediction: Tensor, target: Tensor) -> Tensor:
    pred_norm = prediction.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    cosine = (prediction * target).sum(dim=-1) / (
        pred_norm * target_norm
    ).clamp_min(1e-6)
    both_nonzero = (pred_norm > 1e-6) & (target_norm > 1e-6)
    return torch.where(both_nonzero, 1.0 - cosine.clamp(-1.0, 1.0), torch.ones_like(cosine))


def completion_set_loss(
    validity_logits: Tensor,
    normalized_delta: Tensor,
    target_delta: Tensor,
    target_valid: Tensor,
    endpoint_valid: Tensor,
    *,
    class_weight: float = 1.0,
    coordinate_weight: float = 5.0,
    direction_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Permutation-invariant Q=3 loss supporting zero through three targets.

    The small fixed set is matched by exhaustive enumeration (at most six
    assignments), avoiding a SciPy/runtime dependency.  Matching decisions are
    discrete, while the selected classification, coordinate, and direction
    terms remain differentiable.
    """

    if validity_logits.ndim != 3 or validity_logits.shape[-1] != 3:
        raise ValueError("validity_logits must have shape [B,N,3]")
    if normalized_delta.shape != validity_logits.shape + (2,):
        raise ValueError("normalized_delta must have shape [B,N,3,2]")
    if target_delta.shape != normalized_delta.shape:
        raise ValueError("target_delta must match normalized_delta")
    if target_valid.shape != validity_logits.shape:
        raise ValueError("target_valid must have shape [B,N,3]")
    if endpoint_valid.shape != validity_logits.shape[:2]:
        raise ValueError("endpoint_valid must have shape [B,N]")

    # Every ordering of the three target slots is aligned with the three fixed
    # prediction slots.  With M<3 targets, the invalid slots act as no-object
    # labels, so these six orderings also enumerate every injective M-to-Q
    # assignment.  This keeps matching fully vectorized on the accelerator.
    permutations = torch.tensor(
        list(itertools.permutations(range(3))),
        dtype=torch.long,
        device=target_delta.device,
    )
    permuted_targets = target_delta[:, :, permutations, :].permute(2, 0, 1, 3, 4)
    permuted_valid = target_valid[:, :, permutations].permute(2, 0, 1, 3).bool()
    expanded_logits = validity_logits.unsqueeze(0).expand(6, -1, -1, -1)
    expanded_predictions = normalized_delta.unsqueeze(0).expand(6, -1, -1, -1, -1)
    class_sums = F.binary_cross_entropy_with_logits(
        expanded_logits,
        permuted_valid.to(dtype=validity_logits.dtype),
        reduction="none",
    ).sum(dim=-1)
    coordinate_sums = (
        (expanded_predictions - permuted_targets).abs()
        * permuted_valid[..., None]
    ).sum(dim=(-1, -2))
    direction_sums = (
        _safe_direction_cost(expanded_predictions, permuted_targets)
        * permuted_valid
    ).sum(dim=-1)
    matching_cost = (
        class_weight * class_sums
        + coordinate_weight * coordinate_sums
        + direction_weight * direction_sums
    )
    best_permutation = matching_cost.detach().argmin(dim=0, keepdim=True)
    selected_class = class_sums.gather(0, best_permutation).squeeze(0)
    selected_coordinate = coordinate_sums.gather(0, best_permutation).squeeze(0)
    selected_direction = direction_sums.gather(0, best_permutation).squeeze(0)

    endpoint_mask = endpoint_valid.to(dtype=validity_logits.dtype)
    matched_per_endpoint = target_valid.sum(dim=-1).to(dtype=validity_logits.dtype)
    endpoint_denominator = (endpoint_mask.sum() * 3.0).clamp_min(1.0)
    matched_count = (matched_per_endpoint * endpoint_mask).sum()
    class_loss = (selected_class * endpoint_mask).sum() / endpoint_denominator
    coordinate_loss = (
        (selected_coordinate * endpoint_mask).sum()
        / (matched_count * 2.0).clamp_min(1.0)
    )
    direction_loss = (
        (selected_direction * endpoint_mask).sum() / matched_count.clamp_min(1.0)
    )
    total = (
        class_weight * class_loss
        + coordinate_weight * coordinate_loss
        + direction_weight * direction_loss
    )
    return {
        "loss": total,
        "loss_class": class_loss,
        "loss_coordinate": coordinate_loss,
        "loss_direction": direction_loss,
        "matched_targets": matched_count.detach(),
    }


def permutation_invariant_set_loss(
    valid_logits: Tensor,
    deltas: Tensor,
    target_valid: Tensor,
    target_deltas: Tensor,
    query_valid: Tensor,
    *,
    class_weight: float = 1.0,
    coordinate_weight: float = 5.0,
    direction_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Stable training API with explicit component names.

    The compatibility keys from :func:`completion_set_loss` are intentionally
    not exposed here; callers receive ``total/class/coord/direction``.
    """

    losses = completion_set_loss(
        valid_logits,
        deltas,
        target_deltas,
        target_valid,
        query_valid,
        class_weight=class_weight,
        coordinate_weight=coordinate_weight,
        direction_weight=direction_weight,
    )
    return {
        "total": losses["loss"],
        "class": losses["loss_class"],
        "coord": losses["loss_coordinate"],
        "direction": losses["loss_direction"],
        "matched_targets": losses["matched_targets"],
    }


@dataclass(frozen=True)
class ExpansionResult:
    """Graph state and per-round expansion records."""

    nodes: np.ndarray
    edges: np.ndarray
    rounds: tuple[Mapping[str, object], ...]

    @property
    def audit(self) -> tuple[Mapping[str, object], ...]:
        """Alias exposing the per-round audit under its semantic name."""

        return self.rounds


PredictionFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int],
    tuple[np.ndarray, np.ndarray],
]


def _validated_graph(
    nodes: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    node_array = np.asarray(nodes, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    if node_array.ndim != 2 or node_array.shape[1:] != (2,):
        raise ValueError("nodes must have shape [V,2]")
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1:] != (2,):
        raise ValueError("edges must have shape [E,2]")
    if edge_array.size and (edge_array.min() < 0 or edge_array.max() >= len(node_array)):
        raise ValueError("edge index outside node range")
    canonical: set[tuple[int, int]] = set()
    for source, target in edge_array.tolist():
        if source == target:
            raise ValueError("input graph contains a self-loop")
        canonical.add(tuple(sorted((int(source), int(target)))))
    canonical_edges = np.asarray(sorted(canonical), dtype=np.int64)
    if not canonical:
        canonical_edges = np.empty((0, 2), dtype=np.int64)
    return node_array.copy(), canonical_edges


def degree_one_endpoints(
    nodes: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return degree-one node ids and their outward continuation directions."""

    node_array, edge_array = _validated_graph(nodes, edges)
    neighbors: list[list[int]] = [[] for _ in range(len(node_array))]
    for source, target in edge_array.tolist():
        neighbors[source].append(target)
        neighbors[target].append(source)
    endpoint_ids = np.asarray(
        [index for index, adjacent in enumerate(neighbors) if len(adjacent) == 1],
        dtype=np.int64,
    )
    directions = np.empty((len(endpoint_ids), 2), dtype=np.float64)
    for output_index, node_id in enumerate(endpoint_ids.tolist()):
        neighbor_id = neighbors[node_id][0]
        direction = node_array[node_id] - node_array[neighbor_id]
        norm = float(np.linalg.norm(direction))
        directions[output_index] = direction / norm if norm > 1e-12 else 0.0
    return endpoint_ids, directions
