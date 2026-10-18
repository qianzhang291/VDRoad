"""Export fixed-order FP32 SAM-Road features for one 400 x 400 RGB image."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import json

import numpy as np
from vdroad.features import export_patch_features
from vdroad.runtime import load_frozen_base, load_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--samroad-root", type=Path, required=True)
    parser.add_argument("--baseline-config", "--samroad-config", dest="baseline_config", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int,
                        help="Override the configured inference batch size (default: 1)")
    args = parser.parse_args()
    if (args.batch_size is not None and args.batch_size < 1) or args.output.exists():
        parser.error("Use a positive batch size and a new output path")
    config = json.loads(args.config.read_text())
    config["paths"].update(external_samroad_root=str(args.samroad_root.resolve()),
                           baseline_config=str(args.baseline_config.resolve()),
                           sam_checkpoint=str(args.sam_checkpoint.resolve()))
    runtime = load_runtime(config)
    model = load_frozen_base(runtime, args.base_checkpoint, device=args.device)
    image = runtime["read_rgb_img"](str(args.image))
    cache = export_patch_features(model, image, runtime, device=args.device,
                                  batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **cache)
    print(f"Saved {len(cache['features'])} feature patches to {args.output}")


if __name__ == "__main__":
    main()
