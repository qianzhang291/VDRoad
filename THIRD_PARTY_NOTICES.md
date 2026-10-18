# Third-Party Notices

Original repository code and scripts are provided under the [MIT License](LICENSE).
External upstream repositories and runtime dependencies retain their respective
copyright notices and licenses as summarized below.

| Component | Upstream reference | Role and license |
|---|---|---|
| SAM-Road | [`htcr/sam_road`](https://github.com/htcr/sam_road) | External frozen visual encoder and graph extraction framework (MIT License, copyright htcr 2024) |
| Segment Anything (SAM) | [`facebookresearch/segment-anything`](https://github.com/facebookresearch/segment-anything) | Upstream ViT-B image encoder dependency of SAM-Road (Apache License 2.0) |
| DeH4R | [`godx-7/DeH4R`](https://github.com/godx-7/DeH4R) ([arXiv:2508.13669](https://arxiv.org/abs/2508.13669)) | Related iterative graph expansion reference (no DeH4R code or weights are distributed in this repository) |

Runtime Python packages listed in `pyproject.toml` and `requirements-samroad.txt`
(including PyTorch, NumPy, SciPy, scikit-image, NetworkX, Rtree, Shapely, and
`igraph`) are external dependencies installed separately under their respective
licenses. SpaceNet imagery, ground-truth graphs, and pretrained model checkpoints
are not redistributed in this repository.
