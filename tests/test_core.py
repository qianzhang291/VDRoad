from __future__ import annotations
import unittest
import numpy as np
import torch
from vdroad.core import EndpointCompletionHead, completion_set_loss, permutation_invariant_set_loss, sample_endpoint_rois

class EndpointHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.head = EndpointCompletionHead(dropout=0.0)

    def test_shape_range_invalid_mask_and_finite_gradient(self) -> None:
        features = torch.randn(2, 256, 16, 16, requires_grad=True)
        points = torch.tensor(
            [[[0.0, 0.0], [127.0, 127.0], [255.0, 255.0]],
             [[32.0, 224.0], [160.0, 80.0], [200.0, 200.0]]]
        )
        direction = torch.randn(2, 3, 2)
        valid = torch.tensor([[True, True, False], [True, False, True]])
        logits, delta = self.head(
            features, points, direction, valid, mode="full"
        )
        self.assertEqual(tuple(logits.shape), (2, 3, 3))
        self.assertEqual(tuple(delta.shape), (2, 3, 3, 2))
        self.assertTrue(bool((delta.abs() <= 1.0).all()))
        self.assertTrue(torch.equal(logits[~valid], torch.zeros_like(logits[~valid])))
        self.assertTrue(torch.equal(delta[~valid], torch.zeros_like(delta[~valid])))
        (logits.sum() + delta.sum()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for parameter in self.head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_ablation_modes_use_identical_parameters(self) -> None:
        parameter_names = tuple(name for name, _value in self.head.named_parameters())
        parameter_count = sum(value.numel() for value in self.head.parameters())
        features = torch.randn(1, 256, 8, 8)
        points = torch.tensor([[[31.0, 31.0]]])
        direction = torch.tensor([[[1.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        outputs = {}
        for mode in ("full", "visual_only", "direction_only"):
            outputs[mode] = self.head(
                features, points, direction, valid, mode=mode
            )
            self.assertEqual(
                tuple(name for name, _value in self.head.named_parameters()),
                parameter_names,
            )
            self.assertEqual(sum(value.numel() for value in self.head.parameters()), parameter_count)
        self.assertFalse(torch.allclose(outputs["full"][0], outputs["visual_only"][0]))
        self.assertFalse(torch.allclose(outputs["full"][0], outputs["direction_only"][0]))

    def test_empty_endpoint_set_is_supported(self) -> None:
        logits, delta = self.head(
            torch.randn(2, 256, 8, 8),
            torch.empty(2, 0, 2),
            torch.empty(2, 0, 2),
            torch.empty(2, 0, dtype=torch.bool),
        )
        self.assertEqual(tuple(logits.shape), (2, 0, 3))
        self.assertEqual(tuple(delta.shape), (2, 0, 3, 2))

    def test_roi_rotation_and_boundaries(self) -> None:
        feature = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
        points = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [6.0, 6.0]]])
        roi = sample_endpoint_rois(
            feature, points, patch_size=7, roi_size=5, roi_radius_pixels=2.0
        )
        self.assertEqual(tuple(roi.shape), (1, 3, 1, 5, 5))
        self.assertTrue(torch.isfinite(roi).all())
        self.assertEqual(float(roi[0, 1, 0, 2, 2]), float(feature[0, 0, 0, 0]))
        self.assertEqual(float(roi[0, 2, 0, 2, 2]), float(feature[0, 0, 6, 6]))

        rotated_feature = torch.rot90(feature, 1, dims=(-2, -1))
        rotated_points = points.clone()
        rotated_points[..., 0] = points[..., 1]
        rotated_points[..., 1] = 6.0 - points[..., 0]
        rotated_roi = sample_endpoint_rois(
            rotated_feature,
            rotated_points,
            patch_size=7,
            roi_size=5,
            roi_radius_pixels=2.0,
        )
        self.assertTrue(
            torch.allclose(rotated_roi, torch.rot90(roi, 1, dims=(-2, -1)), atol=1e-5)
        )

class SetLossTests(unittest.TestCase):
    def test_stable_loss_api_names(self) -> None:
        logits = torch.zeros(1, 1, 3, requires_grad=True)
        deltas = torch.zeros(1, 1, 3, 2, requires_grad=True)
        result = permutation_invariant_set_loss(
            logits,
            deltas,
            torch.tensor([[[True, False, False]]]),
            torch.tensor([[[[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertEqual(
            set(result), {"total", "class", "coord", "direction", "matched_targets"}
        )
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_permutation_invariance_for_zero_to_three_targets(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(1, 4, 3, requires_grad=True)
        prediction_raw = torch.randn(1, 4, 3, 2, requires_grad=True)
        prediction = torch.tanh(prediction_raw)
        targets = torch.tensor(
            [[
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.4, 0.1], [0.0, 0.0], [0.0, 0.0]],
                [[0.2, 0.7], [-0.5, 0.3], [0.0, 0.0]],
                [[0.1, 0.8], [-0.7, 0.2], [0.4, -0.6]],
            ]],
            dtype=torch.float32,
        )
        target_valid = torch.tensor(
            [[[False, False, False], [True, False, False], [True, True, False], [True, True, True]]]
        )
        endpoint_valid = torch.ones(1, 4, dtype=torch.bool)
        first = completion_set_loss(logits, prediction, targets, target_valid, endpoint_valid)
        permutation = torch.tensor([2, 0, 1])
        second = completion_set_loss(
            logits,
            prediction,
            targets[:, :, permutation],
            target_valid[:, :, permutation],
            endpoint_valid,
        )
        for name in ("loss", "loss_class", "loss_coordinate", "loss_direction"):
            self.assertTrue(torch.allclose(first[name], second[name], atol=1e-6), name)
        first["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(prediction_raw.grad).all())

    def test_all_invalid_endpoints_still_has_finite_zero_gradient(self) -> None:
        logits = torch.randn(2, 2, 3, requires_grad=True)
        prediction = torch.randn(2, 2, 3, 2, requires_grad=True)
        result = completion_set_loss(
            logits,
            prediction,
            torch.zeros_like(prediction),
            torch.zeros_like(logits, dtype=torch.bool),
            torch.zeros(2, 2, dtype=torch.bool),
        )
        self.assertEqual(float(result["loss"].detach()), 0.0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(prediction.grad).all())
