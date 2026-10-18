"""Apply the fixed completion head to a graph and cached visual features."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import json

import numpy as np

from vdroad.inference import complete_graph, load_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["full", "visual_only", "direction_only"], default="full")
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        parser.error("Output already exists; choose a new output path")
    config = json.loads(args.config.read_text())
    spec = config["inference"]
    with np.load(args.graph, allow_pickle=False) as graph, np.load(args.features, allow_pickle=False) as cache:
        result = complete_graph(
            graph["nodes_xy"], graph["edges"], cache["features"],
            cache["origins_xy"], cache["image_size_hw"],
            load_head(args.checkpoint, args.device, args.mode, config=config), mode=args.mode,
            rounds=int(spec["rounds"]), score_threshold=float(spec["score_threshold"]),
            merge_distance=float(spec["merge_distance"]),
            same_direction_cosine=float(spec["same_direction_cosine"]),
            max_step_pixels=float(spec["max_step_pixels"]),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, nodes_xy=result.nodes, edges=result.edges)
    with audit_path.open("x") as stream:
        json.dump({"mode": args.mode, "rounds": result.audit}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result.nodes)} nodes and {len(result.edges)} edges to {args.output}")


if __name__ == "__main__":
    main()
