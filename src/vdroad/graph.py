"""Deterministic two-phase graph expansion for endpoint completion."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .core import (
    ExpansionResult,
    PredictionFunction,
    _validated_graph,
    degree_one_endpoints,
)


def expand_graph(
    nodes_xy: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
    predict_callback: PredictionFunction,
    *,
    rounds: int,
    score_threshold: float,
    merge_distance: float,
    same_direction_cosine: float = 0.94,
    mode: str = "learned_dynamic",
    bounds_xy: tuple[float, float] | None = None,
    minimum_edge_length: float = 1.0,
) -> ExpansionResult:
    """Expand from a frozen round-start graph, then resolve candidates by score.

    Candidate generation is independent of graph mutation.  Conflict
    resolution uses ``(-score, source, target_x, target_y)`` ordering, so
    permuting the unordered Q slots does not change the result.  Snap targets
    are restricted to round-start vertices and explicitly exclude the source.
    """
    if rounds < 0 or merge_distance < 0 or minimum_edge_length <= 0:
        raise ValueError("invalid expansion geometry")
    if mode not in ("learned_dynamic", "learned_snap"):
        raise ValueError(f"unsupported inference mode: {mode}")
    nodes, input_edges = _validated_graph(nodes_xy, edges)
    edge_set = {tuple(edge) for edge in input_edges.tolist()}
    audits = []
    for round_index in range(rounds):
        round_nodes = nodes.copy()
        round_edges = np.asarray(sorted(edge_set), dtype=np.int64)
        if not edge_set:
            round_edges = np.empty((0, 2), dtype=np.int64)
        endpoint_ids, incoming = degree_one_endpoints(round_nodes, round_edges)
        audit = {
            "round": round_index + 1, "endpoint_ids": endpoint_ids.tolist(),
            "candidate_count": 0, "added_nodes": [], "added_edges": [],
            "merges": [], "rejected": {}, "events": [],
            "resolution": "round_start_candidates_sorted_by_score_source_target",
        }
        if not len(endpoint_ids):
            audits.append(audit)
            break
        scores, deltas = predict_callback(
            endpoint_ids.copy(), round_nodes[endpoint_ids].copy(), incoming.copy(), round_index
        )
        scores, deltas = np.asarray(scores, dtype=np.float64), np.asarray(deltas, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != len(endpoint_ids):
            raise ValueError("predictor scores must have shape [E,Q]")
        if deltas.shape != scores.shape + (2,):
            raise ValueError("predictor deltas must have shape [E,Q,2]")
        audit["candidate_count"] = int(scores.size)
        candidates = []

        def reject(reason: str, event: dict) -> None:
            audit["rejected"][reason] = int(audit["rejected"].get(reason, 0)) + 1
            event.update({"action": "reject", "reason": reason})
            audit["events"].append(event)

        for endpoint_offset, source in enumerate(endpoint_ids.tolist()):
            for query in range(scores.shape[1]):
                score = float(scores[endpoint_offset, query])
                delta = deltas[endpoint_offset, query]
                event = {"source": source, "query": query, "score": score, "delta": delta.tolist()}
                if not np.isfinite(score) or not np.isfinite(delta).all():
                    reject("non_finite", event); continue
                if score < score_threshold:
                    reject("below_threshold", event); continue
                length = float(np.linalg.norm(delta))
                if length < minimum_edge_length:
                    reject("below_minimum_edge_length", event); continue
                target = round_nodes[source] + delta
                event["target_xy"] = target.tolist()
                if bounds_xy is not None and (
                    target[0] < 0 or target[1] < 0 or target[0] > bounds_xy[0] - 1 or target[1] > bounds_xy[1] - 1
                ):
                    reject("out_of_bounds", event); continue
                candidates.append((score, source, query, target, delta, event))
        candidates.sort(key=lambda row: (-row[0], row[1], float(row[3][0]), float(row[3][1]), row[2]))

        accepted_targets: list[np.ndarray] = []
        for score, source, query, target, delta, event in candidates:
            length = float(np.linalg.norm(delta))
            unit = delta / length
            adjacent = []
            for left, right in edge_set:
                if left == source:
                    adjacent.append(right)
                elif right == source:
                    adjacent.append(left)
            duplicate_direction = any(
                np.linalg.norm(nodes[neighbor] - nodes[source]) > 1e-9
                and float(np.dot(unit, (nodes[neighbor] - nodes[source]) /
                                 np.linalg.norm(nodes[neighbor] - nodes[source]))) >= same_direction_cosine
                for neighbor in adjacent
            )
            if duplicate_direction:
                reject("existing_direction", event); continue
            if any(float(np.linalg.norm(previous - target)) <= merge_distance for previous in accepted_targets):
                reject("round_candidate_conflict", event); continue

            distances = np.linalg.norm(round_nodes - target[None, :], axis=1)
            snap_target = None
            for nearest in np.argsort(distances).tolist():
                if float(distances[nearest]) > merge_distance:
                    break
                if nearest == source:
                    continue
                edge = tuple(sorted((source, int(nearest))))
                if edge in edge_set:
                    continue
                snap_target = int(nearest)
                break
            if snap_target is not None:
                edge = tuple(sorted((source, snap_target)))
                edge_set.add(edge)
                accepted_targets.append(round_nodes[snap_target].copy())
                event.update({"action": "merge", "target": snap_target, "edge": list(edge)})
                audit["merges"].append({"source": source, "target": snap_target})
                audit["added_edges"].append(list(edge))
                audit["events"].append(event)
                continue
            if mode == "learned_snap":
                reject("snap_no_legal_round_start_target", event); continue
            new_id = len(nodes)
            nodes = np.concatenate([nodes, target[None, :]], axis=0)
            edge = tuple(sorted((source, new_id)))
            edge_set.add(edge)
            accepted_targets.append(target.copy())
            event.update({"action": "insert", "target": new_id, "edge": list(edge)})
            audit["added_nodes"].append({"id": new_id, "xy": target.tolist()})
            audit["added_edges"].append(list(edge))
            audit["events"].append(event)
        audits.append(audit)
    final_edges = np.asarray(sorted(edge_set), dtype=np.int64) if edge_set else np.empty((0, 2), dtype=np.int64)
    return ExpansionResult(nodes.copy(), final_edges, tuple(audits))


def simple_direction_completion(
    nodes: np.ndarray | Sequence[Sequence[float]],
    edges: np.ndarray | Sequence[Sequence[int]],
    *, rounds: int, step_pixels: float, merge_distance: float,
    same_direction_cosine: float = 0.94, mode: str = "learned_dynamic",
    bounds_xy: tuple[float, float] | None = None,
) -> ExpansionResult:
    if step_pixels <= 0:
        raise ValueError("step_pixels must be positive")
    def predictor(_ids, _xy, incoming, _round):
        return np.ones((len(incoming), 1)), incoming[:, None, :] * float(step_pixels)
    return expand_graph(
        nodes, edges, predictor, rounds=rounds, score_threshold=0.5,
        merge_distance=merge_distance, same_direction_cosine=same_direction_cosine,
        mode=mode, bounds_xy=bounds_xy,
    )


__all__ = ["expand_graph", "simple_direction_completion"]
