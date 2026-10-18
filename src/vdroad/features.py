"""FP32 visual-feature export using the configured SAM-Road patch schedule."""
from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def export_patch_features(model, image, runtime, *, device="cpu", batch_size=None):
    if image is None or image.shape != (400, 400, 3):
        raise ValueError("Expected a 400 x 400 RGB image")
    spec = runtime["sam_config"]
    batch_size = int(spec.INFER_BATCH_SIZE if batch_size is None else batch_size)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    patches = runtime["get_patch_info_one_img"](
        0, image.shape[0], spec.SAMPLE_MARGIN, spec.PATCH_SIZE, spec.INFER_PATCHES_PER_EDGE,
    )
    expected = int(spec.INFER_PATCHES_PER_EDGE) ** 2
    if len(patches) != expected:
        raise ValueError(f"Expected {expected} feature patches, received {len(patches)}")
    features = []
    for start in range(0, len(patches), batch_size):
        arrays = [image[y0:y1, x0:x1, :]
                  for _, (x0, y0), (x1, y1) in patches[start:start + batch_size]]
        pixels = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
        _mask, embedding = model.infer_masks_and_img_features(pixels)
        features.append(embedding.float().cpu().numpy())
    return {
        "features": np.concatenate(features),
        "origins_xy": np.asarray([patch[1] for patch in patches], dtype=np.int64),
        "image_size_hw": np.asarray(image.shape[:2], dtype=np.int64),
    }
