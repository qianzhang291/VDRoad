# VDRoad

**Visual–Directional Road Graph Completion**

PyTorch implementation of **VDRoad**, a lightweight graph-level completion module for road network extraction from aerial imagery.

Given an initial road graph with dangling (degree-one) endpoints and frozen visual encoder features from [SAM-Road](https://github.com/htcr/sam_road), the completion head conditions on each endpoint's incoming edge direction and local visual ROI features to predict candidate continuation branches, iteratively expanding and merging the road network graph over three rounds.

---

## Repository Structure

```text
src/vdroad/  Completion head, set loss, two-phase graph expansion, inference, and trainer
scripts/                  Command-line entry points for training, feature export, and graph completion
configs/                  Default configuration (default.json) and scene split manifests
tests/                    Unit tests for completion head, set loss, rotation equivariance, and expansion
```

---

## Installation

Core package requirements: Python $\ge 3.10$, NumPy, and PyTorch $\ge 2.7$.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For training and visual feature extraction with an external [SAM-Road](https://github.com/htcr/sam_road) checkout on Python 3.12, install the integration dependencies:

```bash
python -m pip install -r requirements-samroad.txt
```

---

## Dataset Preparation

Follow the [SAM-Road data preparation guide](https://github.com/htcr/sam_road#data-preparation) to prepare the **SpaceNet** road dataset (`RGB_1.0_meter` directory):

```text
/path/to/RGB_1.0_meter/
├── <scene_id>__rgb.png       # 400×400 RGB aerial imagery (1 m/px)
└── <scene_id>__gt_graph.p    # Ground-truth road graph (pickle adjacency dictionary)
```

Scene split manifests are provided in [`configs/splits/`](configs/splits/):
- [`configs/splits/outer.json`](configs/splits/outer.json): Scene role assignment (`fit`: 1,427 scenes; `dev`: 306 scenes; `cal`: 305 scenes).
- [`configs/splits/inner.json`](configs/splits/inner.json): Disjoint split of the 1,427 `fit` scenes into `roles.train` (1,284 scenes) and `roles.holdout` (143 scenes).

---

## Usage

### 1. Train Completion Head

Train the `full` completion head on top of a frozen SAM-Road checkpoint:

```bash
python scripts/train.py \
  --config configs/default.json \
  --samroad-root /path/to/sam_road \
  --samroad-config /path/to/toponet_vitb_256_spacenet.yaml \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --base-checkpoint /path/to/samroad_base.pt \
  --data-root /path/to/RGB_1.0_meter \
  --train-split configs/splits/inner.json \
  --holdout-split configs/splits/inner.json \
  --output-dir outputs/seed20260916 \
  --seed 20260916
```

Training runs all 20 epochs at batch size 96 with BF16 mixed precision, and
selects the minimum inner-holdout loss, keeping the earlier epoch on ties.
The same 4,096 unrotated holdout patches are reused each epoch. Outputs are
`best_full.pt`, `last.pt` (for `--resume`), and training curves in `--output-dir`.
For input ablations, explicitly pass `--modes full visual_only direction_only`.

Both split arguments intentionally use `inner.json`: the trainer reads
`roles.train` for training and `roles.holdout` for model selection. These scene
sets are disjoint; the example does not train and select on the same scenes.

### 2. Export Frozen Visual Features

Export the $8 \times 8$ grid of overlapping $256 \times 256$ patch features (`features`, `origins_xy`, `image_size_hw`) for a $400 \times 400$ scene. The configured inference batch size is 1 and features are stored in FP32:

```bash
python scripts/export_features.py \
  --config configs/default.json \
  --image /path/to/scene__rgb.png \
  --base-checkpoint /path/to/samroad_base.pt \
  --samroad-root /path/to/sam_road \
  --samroad-config /path/to/toponet_vitb_256_spacenet.yaml \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --output features.npz
```

### 3. Complete a Road Graph

Run three-round iterative endpoint completion on an initial graph stored as an `.npz` archive (`nodes_xy` of shape `[N, 2]` in pixel `(x, y)` coordinates and `edges` of shape `[M, 2]` with integer node indices):

```bash
python scripts/complete_graph.py \
  --config configs/default.json \
  --graph initial_graph.npz \
  --features features.npz \
  --checkpoint outputs/seed20260916/best_full.pt \
  --output completed_graph.npz \
  --device cpu
```

---

## Acknowledgements and License

This work builds on [SAM-Road](https://github.com/htcr/sam_road) and [Segment Anything](https://github.com/facebookresearch/segment-anything), with iterative graph expansion inspired by [DeH4R](https://arxiv.org/abs/2508.13669).

Released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external dependency notices.
