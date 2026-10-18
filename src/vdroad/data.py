"""Training-split dataset and target generation utilities for endpoint completion.

Targets are generated from SAM-Road SpaceNet ground-truth road graphs: one incident
edge is treated as the observed incoming edge, and remaining branches are traced to
a bounded target distance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DATE = "20260916"
INNER_HOLDOUT_BY_CITY = {
    "AOI_2_Vegas": 55,
    "AOI_3_Paris": 14,
    "AOI_4_Shanghai": 58,
    "AOI_5_Khartoum": 16,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def scene_city(scene_id: str) -> str:
    for city in INNER_HOLDOUT_BY_CITY:
        if scene_id.startswith(city + "_"):
            return city
    raise ValueError(f"unrecognized SpaceNet scene id: {scene_id}")


def build_inner_split(fit_ids: Sequence[str], salt: str) -> dict[str, Any]:
    """Create a deterministic, per-city 90/10 split of the frozen FIT role."""
    require(len(fit_ids) == len(set(fit_ids)) == 1427, "FIT1427 changed")
    grouped = {city: [] for city in INNER_HOLDOUT_BY_CITY}
    for scene_id in fit_ids:
        grouped[scene_city(scene_id)].append(scene_id)
    holdout: list[str] = []
    allocation: dict[str, dict[str, int]] = {}
    for city, ids in grouped.items():
        ranked = sorted(
            ids,
            key=lambda value: (
                hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest(),
                value,
            ),
        )
        count = INNER_HOLDOUT_BY_CITY[city]
        holdout.extend(ranked[:count])
        allocation[city] = {"train": len(ids) - count, "holdout": count}
    holdout_set = set(holdout)
    train = [scene_id for scene_id in fit_ids if scene_id not in holdout_set]
    holdout = [scene_id for scene_id in fit_ids if scene_id in holdout_set]
    require(len(train) == 1284 and len(holdout) == 143, "inner split counts changed")
    require(not (set(train) & set(holdout)), "inner split overlaps")
    return {
        "schema_version": 1,
        "date": DATE,
        "salt": salt,
        "source_role": "fit",
        "allocation_by_city": allocation,
        "roles": {"train": train, "holdout": holdout},
    }


def make_schedule(seed: int, scene_count: int, sample_count: int) -> dict[str, np.ndarray]:
    require(scene_count > 0 and sample_count > 0, "invalid schedule size")
    rng = np.random.RandomState(int(seed))
    return {
        "scene_index": rng.randint(0, scene_count, size=sample_count).astype(np.uint16),
        "x0": rng.randint(0, 145, size=sample_count).astype(np.uint8),
        "y0": rng.randint(0, 145, size=sample_count).astype(np.uint8),
        "rotation": rng.randint(0, 4, size=sample_count).astype(np.uint8),
        "graph_seed": rng.randint(0, 2**32 - 1, size=sample_count, dtype=np.uint32),
    }


def schedule_sha256(schedule: Mapping[str, np.ndarray]) -> dict[str, str]:
    result = {}
    for name, value in schedule.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


def rotate_xy(points: np.ndarray, rotation: int, patch_size: int = 256) -> np.ndarray:
    """Match np.rot90 around the discrete pixel center (N-1)/2."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 2)
    center = 0.5 * (patch_size - 1)
    rotation_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    return (points - center) @ np.linalg.matrix_power(rotation_matrix.T, int(rotation)) + center


def rotate_vectors(vectors: np.ndarray, rotation: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors.reshape(-1, 2)
    rotation_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    return vectors @ np.linalg.matrix_power(rotation_matrix.T, int(rotation))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def _trace_branch(graph: Any, points: np.ndarray, current: int, first: int,
                  target_distance: float, max_distance: float) -> np.ndarray:
    """Trace one branch, choosing the straightest continuation at later junctions."""
    origin = points[current].astype(np.float32)
    previous, node = int(current), int(first)
    travelled = float(np.linalg.norm(points[node] - points[previous]))
    while travelled < target_distance:
        candidates = [int(value) for value in graph.neighbors(node) if int(value) != previous]
        if not candidates:
            break
        incoming = _unit(points[node] - points[previous])
        next_node = max(
            candidates,
            key=lambda value: (
                float(np.dot(incoming, _unit(points[value] - points[node]))),
                -int(value),
            ),
        )
        segment = float(np.linalg.norm(points[next_node] - points[node]))
        if travelled + segment > max_distance:
            remaining = max_distance - travelled
            if remaining <= 0:
                break
            direction = _unit(points[next_node] - points[node])
            return (points[node] + remaining * direction - origin).astype(np.float32)
        travelled += segment
        previous, node = node, next_node
        if graph.degree(node) != 2 and travelled >= target_distance:
            break
    delta = points[node] - origin
    length = float(np.linalg.norm(delta))
    if length > max_distance:
        delta = delta * (max_distance / length)
    return delta.astype(np.float32)


def endpoint_targets(graph: Any, points: np.ndarray, vertex: int, previous: int,
                     target_distance: float, max_distance: float,
                     max_targets: int) -> np.ndarray:
    outgoing = [int(value) for value in graph.neighbors(vertex) if int(value) != int(previous)]
    targets = [
        _trace_branch(graph, points, int(vertex), neighbor, target_distance, max_distance)
        for neighbor in outgoing
    ]
    targets = [value for value in targets if float(np.linalg.norm(value)) >= 1.0]
    if len(targets) > max_targets:
        incoming = _unit(points[vertex] - points[previous])
        targets = sorted(
            targets,
            key=lambda value: (-float(np.dot(incoming, _unit(value))), float(np.linalg.norm(value))),
        )[:max_targets]
    return np.asarray(targets, dtype=np.float32).reshape(-1, 2)


class EndpointFitCorpus(Dataset):
    """Fixed-shape patch/query corpus restricted to an explicit FIT subset."""

    def __init__(self, *, config: Mapping[str, Any], scene_ids: Sequence[str],
                 schedule: Mapping[str, np.ndarray], runtime: Mapping[str, Any],
                 samples_per_epoch: int, max_epochs: int) -> None:
        manifest_path = Path(config["paths"]["prospective_manifest"])
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        fit = set(json.loads(manifest_path.read_text(encoding="utf-8"))["roles"]["fit"])
        require(set(scene_ids) <= fit, "endpoint corpus contains a non-FIT scene")
        require(len(scene_ids) == len(set(scene_ids)), "endpoint scene list contains duplicates")
        require(all(len(value) >= samples_per_epoch * max_epochs for value in schedule.values()),
                "endpoint schedule is too short")
        self.config = config
        self.scene_ids = list(scene_ids)
        self.schedule = schedule
        self.runtime = runtime
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_epochs = int(max_epochs)
        self.epoch = 0
        spec = config["completion"]
        self.query_count = int(spec["query_count"])
        self.max_targets = int(spec["num_predictions"])
        self.target_distance = float(spec["target_step_pixels"])
        self.max_distance = float(spec["max_step_pixels"])
        self.terminal_fraction = float(spec["terminal_sample_fraction"])
        self.rgbs: list[np.ndarray] = []
        self.generators: list[Any] = []
        data_root = Path(config["paths"]["data_root"])
        for position, scene_id in enumerate(self.scene_ids):
            rgb_path = data_root / f"{scene_id}__rgb.png"
            graph_path = data_root / f"{scene_id}__gt_graph.p"
            require(rgb_path.is_file() and graph_path.is_file(), f"missing FIT assets: {scene_id}")
            rgb = runtime["read_rgb_img"](str(rgb_path))
            require(rgb is not None and rgb.shape == (400, 400, 3), f"invalid FIT RGB: {scene_id}")
            import pickle
            with graph_path.open("rb") as stream:
                raw_graph = pickle.load(stream)
            transform = lambda vertices: np.stack([vertices[:, 1], 400 - vertices[:, 0]], axis=1)
            generator = runtime["GraphLabelGenerator"](runtime["sam_config"], raw_graph, transform)
            self.rgbs.append(rgb)
            self.generators.append(generator)
            if len(self.scene_ids) > 100 and (position + 1) % 200 == 0:
                print(f"Loaded endpoint FIT scenes {position + 1}/{len(self.scene_ids)}", flush=True)

    def set_epoch(self, epoch: int) -> None:
        require(0 <= int(epoch) < self.max_epochs, "endpoint epoch out of range")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _select_vertices(self, generator: Any, x0: int, y0: int,
                         rng: np.random.RandomState) -> list[int]:
        margin = int(np.ceil(self.max_distance + 2.0))
        indices = list(generator.graph_rtee.intersection(
            (x0 + margin, y0 + margin, x0 + 256 - margin, y0 + 256 - margin)
        ))
        graph = generator.full_graph_subdivide
        terminals = [int(value) for value in indices if graph.degree(int(value)) == 1]
        continuations = [int(value) for value in indices if graph.degree(int(value)) >= 2]
        terminal_slots = min(len(terminals), int(round(self.query_count * self.terminal_fraction)))
        chosen: list[int] = []
        if terminal_slots:
            chosen.extend(rng.choice(terminals, size=terminal_slots, replace=len(terminals) < terminal_slots).tolist())
        remaining = self.query_count - len(chosen)
        pool = continuations if continuations else terminals
        if pool and remaining:
            chosen.extend(rng.choice(pool, size=remaining, replace=len(pool) < remaining).tolist())
        rng.shuffle(chosen)
        return chosen

    def _materialize(self, global_index: int) -> dict[str, Tensor]:
        scene_index = int(self.schedule["scene_index"][global_index])
        x0, y0 = int(self.schedule["x0"][global_index]), int(self.schedule["y0"][global_index])
        rotation = int(self.schedule["rotation"][global_index])
        graph_seed = int(self.schedule["graph_seed"][global_index])
        rng = np.random.RandomState(graph_seed)
        rgb = np.rot90(self.rgbs[scene_index][y0:y0 + 256, x0:x0 + 256], rotation, axes=(0, 1)).copy()
        generator = self.generators[scene_index]
        graph = generator.full_graph_subdivide
        points = np.asarray(generator.subdivide_points, dtype=np.float32)
        chosen = self._select_vertices(generator, x0, y0, rng)

        queries = np.zeros((self.query_count, 2), dtype=np.float32)
        incoming = np.zeros((self.query_count, 2), dtype=np.float32)
        targets = np.zeros((self.query_count, self.max_targets, 2), dtype=np.float32)
        target_valid = np.zeros((self.query_count, self.max_targets), dtype=bool)
        query_valid = np.zeros((self.query_count,), dtype=bool)
        for slot, vertex in enumerate(chosen[:self.query_count]):
            neighbors = [int(value) for value in graph.neighbors(vertex)]
            if not neighbors:
                continue
            previous = int(neighbors[int(rng.randint(0, len(neighbors)))])
            clean_query = points[vertex] - np.array([x0, y0], dtype=np.float32)
            noisy_query = clean_query + rng.normal(0.0, 1.0, size=2).astype(np.float32)
            branch_targets = endpoint_targets(
                graph, points, vertex, previous, self.target_distance,
                self.max_distance, self.max_targets,
            )
            outward = _unit(points[vertex] - points[previous]).astype(np.float32)
            rotated_query = rotate_xy(noisy_query[None], rotation)[0]
            rotated_incoming = rotate_vectors(outward[None], rotation)[0]
            correction = clean_query - noisy_query
            rotated_targets = rotate_vectors(branch_targets + correction, rotation)
            count = min(len(rotated_targets), self.max_targets)
            queries[slot] = np.clip(rotated_query, 0.0, 255.0)
            incoming[slot] = _unit(rotated_incoming)
            if count:
                targets[slot, :count] = np.clip(
                    rotated_targets[:count] / self.max_distance, -1.0, 1.0
                )
                target_valid[slot, :count] = True
            query_valid[slot] = True

        return {
            "rgb": torch.tensor(rgb, dtype=torch.float32),
            "query_points": torch.tensor(queries, dtype=torch.float32),
            "incoming_directions": torch.tensor(incoming, dtype=torch.float32),
            "target_deltas": torch.tensor(targets, dtype=torch.float32),
            "target_valid": torch.tensor(target_valid, dtype=torch.bool),
            "query_valid": torch.tensor(query_valid, dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> Mapping[str, Tensor]:
        return self._materialize(self.epoch * self.samples_per_epoch + int(index))


__all__ = [
    "EndpointFitCorpus",
    "build_inner_split",
    "endpoint_targets",
    "make_schedule",
    "require",
    "rotate_vectors",
    "rotate_xy",
    "schedule_sha256",
]
