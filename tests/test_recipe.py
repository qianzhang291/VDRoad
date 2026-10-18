"""Recipe and recovery checks using artificial tensors only."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vdroad import train
from vdroad.features import export_patch_features
from vdroad.inference import load_head
from vdroad.runtime import load_runtime


ROOT = Path(__file__).resolve().parents[1]


class ToyBase(torch.nn.Module):
    def infer_masks_and_img_features(self, rgb):
        return None, torch.zeros(len(rgb), 256, 4, 4)


class ToyCorpus(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.count = kwargs["samples_per_epoch"]

    def set_epoch(self, _epoch):
        pass

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "rgb": torch.zeros(8, 8, 3),
            "query_points": torch.tensor([[128., 128.]]),
            "incoming_directions": torch.tensor([[1., 0.]]),
            "target_deltas": torch.tensor([[[.3, .2], [0., 0.], [0., 0.]]]),
            "target_valid": torch.tensor([[index % 2 == 0, False, False]]),
            "query_valid": torch.tensor([True]),
        }


class RecipeTests(unittest.TestCase):
    def test_fixed_inner_patches_are_identical_across_epochs(self):
        schedule = train.build_schedule(
            seed=20260916 + 300000, role="holdout", scene_count=143,
            samples_per_epoch=16, epochs=3, fixed_holdout=True,
        )
        self.assertTrue(np.all(schedule["rotation"] == 0))
        for values in schedule.values():
            np.testing.assert_array_equal(values[:16], values[16:32])
            np.testing.assert_array_equal(values[:16], values[32:])
        training = train.build_schedule(
            seed=20260916 + 200000, role="train", scene_count=1284,
            samples_per_epoch=16, epochs=3, fixed_holdout=True,
        )
        self.assertFalse(np.array_equal(training["graph_seed"][:16], training["graph_seed"][16:32]))

    def test_configured_8_by_8_feature_order_and_single_patch_batches(self):
        axis = np.linspace(0, 144, 8).astype(int)
        patches = [(0, (int(x), int(y)), (int(x) + 256, int(y) + 256))
                   for y in axis for x in axis]
        calls = []

        class Encoder(torch.nn.Module):
            def infer_masks_and_img_features(self, pixels):
                calls.append(len(pixels))
                value = pixels[:, 0, 0, 0]
                return None, value[:, None, None, None].expand(-1, 256, 2, 2)

        runtime = {"sam_config": SimpleNamespace(
            PATCH_SIZE=256, SAMPLE_MARGIN=0, INFER_PATCHES_PER_EDGE=8, INFER_BATCH_SIZE=1),
            "get_patch_info_one_img": lambda *_args: patches}
        image = np.zeros((400, 400, 3), np.uint8)
        image[:, :, 0] = np.arange(400, dtype=np.uint16)[None] % 256
        result = export_patch_features(Encoder(), image, runtime)
        self.assertEqual(calls, [1] * 64)
        self.assertEqual(result["features"].shape, (64, 256, 2, 2))
        self.assertEqual(result["features"].dtype, np.float32)
        np.testing.assert_array_equal(result["origins_xy"], [p[1] for p in patches])
        np.testing.assert_array_equal(result["features"][:, 0, 0, 0], [p[1][0] for p in patches])

    def test_runtime_applies_release_protocol_to_upstream_defaults(self):
        config = json.loads((ROOT / "configs/default.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dataset", "model", "utils", "graph_utils", "graph_extraction"):
                (root / (name + ".py")).write_text("")
            for name in ("sam.yaml", "sam.pt"):
                (root / name).write_text("")
            config["paths"].update(external_samroad_root=str(root),
                                   baseline_config=str(root / "sam.yaml"),
                                   sam_checkpoint=str(root / "sam.pt"))
            native = SimpleNamespace(PATCH_SIZE=256, DATASET="spacenet",
                                     USE_SAM_DECODER=False, ENCODER_LORA=False,
                                     FOCAL_LOSS=False, INFER_PATCHES_PER_EDGE=16,
                                     INFER_BATCH_SIZE=64)
            modules = {name: SimpleNamespace(__file__=str(root / (name + ".py")))
                       for name in ("dataset", "model", "utils", "graph_utils", "graph_extraction")}
            for name in ("get_patch_info_one_img", "graph_collate_fn", "read_rgb_img"):
                setattr(modules["dataset"], name, None)
            modules["model"].SAMRoad = ToyBase
            modules["utils"].load_config = lambda _path: native
            with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()), \
                    patch("vdroad.runtime.importlib.import_module", side_effect=modules.__getitem__):
                runtime = load_runtime(config)
            self.assertEqual(runtime["sam_config"].INFER_PATCHES_PER_EDGE, 8)
            self.assertEqual(runtime["sam_config"].INFER_BATCH_SIZE, 1)
            self.assertEqual(runtime["sam_config"].ROAD_THRESHOLD, .341)

    def test_full_20_epochs_and_exact_interrupted_resume(self):
        config = json.loads((ROOT / "configs/default.json").read_text())
        self.assertEqual(config["completion"]["max_epochs"], 20)
        self.assertEqual(config["completion"]["min_epochs"], 20)
        config["completion"].update(hidden_dim=16, heads=2, layers=1, ffn_dim=32,
                                     samples_per_epoch=4, holdout_samples=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("base.pt", "sam.pt", "sam.yaml"):
                (root / name).write_bytes(b"artificial test fixture")
            (root / "outer.json").write_text(json.dumps({"roles": {"fit": ["train", "holdout"]}}))
            (root / "inner.json").write_text(json.dumps({"roles": {"train": ["train"], "holdout": ["holdout"]}}))
            config["paths"] = {
                "external_samroad_root": str(root), "baseline_config": str(root / "sam.yaml"),
                "sam_checkpoint": str(root / "sam.pt"), "data_root": str(root),
                "prospective_manifest": str(root / "outer.json"), "inner_manifest": str(root / "inner.json"),
            }
            config_path = root / "recipe.json"
            config_path.write_text(json.dumps(config))
            common = ["--config", str(config_path), "--base-checkpoint", str(root / "base.pt"),
                      "--batch-size", "2", "--workers", "0", "--device", "cpu", "--precision", "fp32"]
            original_save = train.atomic_save

            def interrupt_after_checkpoint(path, value, **kwargs):
                original_save(path, value, **kwargs)
                if path.name == "last.pt" and value["completed_epoch"] == 3:
                    raise InterruptedError("artificial interruption after a complete checkpoint")

            with patch.object(train, "load_runtime", return_value={}), \
                    patch.object(train, "load_frozen_base", side_effect=lambda *_a, **_kw: ToyBase()), \
                    patch.object(train, "EndpointFitCorpus", ToyCorpus), contextlib.redirect_stdout(io.StringIO()):
                train.main(common + ["--output-dir", str(root / "reference")])
                with patch.object(train, "atomic_save", side_effect=interrupt_after_checkpoint):
                    with self.assertRaises(InterruptedError):
                        train.main(common + ["--output-dir", str(root / "resumed")])
                train.main(common + ["--output-dir", str(root / "resumed"), "--resume"])
            reference = torch.load(root / "reference/last.pt", weights_only=False)
            resumed = torch.load(root / "resumed/last.pt", weights_only=False)
            self.assertEqual(resumed["completed_epoch"], 20)
            self.assertEqual(resumed["identity"]["mode_order"], ["full"])
            self.assertEqual(reference["best"], resumed["best"])
            for key, value in reference["head_state_dicts"]["full"].items():
                self.assertTrue(torch.equal(value, resumed["head_state_dicts"]["full"][key]), key)
            selected = load_head(root / "resumed/best_full.pt")
            self.assertEqual(selected.hidden_dim, 16)
            for key, value in resumed["best_payloads"]["full"]["head_state_dict"].items():
                self.assertTrue(torch.equal(value, selected.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
