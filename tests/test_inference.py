import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vdroad.core import EndpointCompletionHead
from vdroad.inference import best_patch_indices, complete_graph, load_head
from vdroad.runtime import load_frozen_base


class PortableInferenceTests(unittest.TestCase):
    def test_base_checkpoint_formats_and_freezing(self):
        class Base(torch.nn.Linear):
            def __init__(self, _config):
                super().__init__(2, 1)

        original = Base(None)
        runtime = {"SAMRoad": Base, "sam_config": SimpleNamespace(PATCH_SIZE=256)}
        formats = {
            "unified": {"state_dict": original.state_dict(), "identity": {"method": "samroad"}},
            "historical": {"model_state_dict": original.state_dict()},
            "plain": original.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in formats.items():
                with self.subTest(format=name):
                    path = Path(directory) / (name + ".pt")
                    torch.save(payload, path)
                    restored = load_frozen_base(runtime, path, device="cpu")
                    for key, value in original.state_dict().items():
                        self.assertTrue(torch.equal(value, restored.state_dict()[key]))
                    self.assertFalse(restored.training)
                    self.assertTrue(all(not p.requires_grad for p in restored.parameters()))

    def test_patch_clearance_and_canonical_tie(self):
        origins = np.asarray([[0, 0], [100, 0], [0, 0]])
        result = best_patch_indices([[150, 100], [200, 100], [50, 50]], origins)
        np.testing.assert_array_equal(result, [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "no covering"):
            best_patch_indices([[399, 399]], origins)

    def test_selected_checkpoint_and_disabled_expansion(self):
        torch.manual_seed(9)
        original = EndpointCompletionHead()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save({"mode": "full", "head_state_dict": original.state_dict()}, path)
            restored = load_head(path)
            self.assertEqual(sum(p.numel() for p in restored.parameters()), 451587)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            with self.assertRaisesRegex(ValueError, "mode"):
                load_head(path, mode="visual_only")
            nodes = np.asarray([[30., 40.], [60., 40.]])
            edges = np.asarray([[0, 1]])
            result = complete_graph(nodes, edges, np.zeros((1, 256, 4, 4), np.float32),
                                    [[0, 0]], [256, 256], restored, rounds=0)
            np.testing.assert_array_equal(result.nodes, nodes)
            np.testing.assert_array_equal(result.edges, edges)

    def test_coordinates_outside_image_are_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "in-image"):
            complete_graph([[-1, 20], [30, 20]], [[0, 1]],
                           np.zeros((1, 256, 4, 4), np.float32), [[0, 0]],
                           [256, 256], EndpointCompletionHead())
