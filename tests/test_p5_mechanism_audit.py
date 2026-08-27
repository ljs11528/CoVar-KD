from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.diagnostics.p5_mechanism_audit import (
    fixed_probe_indices,
    functional_logits,
    gradient_cosine,
    gradient_norm,
    materialize_batch,
    region_alignment_means,
    training_ce_loss,
    updated_parameters,
)


def test_gradient_cosine_and_norm_are_vector_exact():
    left = (torch.tensor([3.0, 4.0]), None)
    right = (torch.tensor([6.0, 8.0]), None)
    assert gradient_norm(left) == pytest.approx(5.0)
    assert gradient_cosine(left, right) == pytest.approx(1.0)


def test_region_alignment_uses_exact_hard_argmax_objective():
    scores = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]
    )
    selection = SimpleNamespace(
        eligible_region_mask=torch.tensor([[[True]]]),
        region_scores=scores,
        selected_region_index=torch.tensor([[[5]]]),
    )
    result = region_alignment_means(selection)
    assert result["fixed"] == pytest.approx(5.0)
    assert result["p4a"] == pytest.approx(6.0)
    assert result["eligible_regions"] == 1


def test_functional_virtual_update_matches_manual_linear_update():
    model = nn.Sequential(nn.Linear(2, 1, bias=False))
    model.eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -1.0]]))
    gradients = (torch.tensor([[0.5, 0.25]]),)
    parameters = updated_parameters(model, gradients, step_size=0.2)
    inputs = torch.tensor([[2.0, 4.0]])
    actual = functional_logits(model, parameters, inputs)
    expected = inputs @ torch.tensor([[0.9, -1.05]]).t()
    assert torch.allclose(actual, expected)
    assert torch.allclose(
        model[0].weight, torch.tensor([[1.0, -1.0]])
    )


def test_fixed_probe_indices_are_deterministic_unique_subset():
    first = fixed_probe_indices(100, 1234, required=24, pool_size=30)
    second = fixed_probe_indices(100, 1234, required=24, pool_size=30)
    assert first == second
    assert len(first) == len(set(first)) == 24
    assert first == sorted(first)


def test_materialize_batch_pads_images_and_ignores_target_border():
    class Dataset:
        files = [{"name": "a"}, {"name": "b"}]

        def __getitem__(self, index):
            if index == 0:
                return (
                    np.ones((3, 2, 3), dtype=np.float32),
                    np.ones((2, 3), dtype=np.int64),
                    None,
                )
            return (
                np.ones((3, 4, 2), dtype=np.float32),
                np.ones((4, 2), dtype=np.int64),
                None,
            )

    batch = materialize_batch(Dataset(), [0, 1], ignore_label=-1)
    assert batch["images"].shape == (2, 3, 4, 3)
    assert batch["targets"].shape == (2, 4, 3)
    assert torch.all(batch["targets"][0, 2:, :] == -1)
    assert torch.all(batch["targets"][1, :, 2] == -1)


def test_deterministic_training_ce_matches_cross_entropy_and_gradient():
    logits = torch.randn(2, 3, 2, 2, requires_grad=True)
    targets = torch.tensor([
        [[0, 1, 2], [1, -1, 0], [2, 2, 1]],
        [[2, 1, 0], [0, 1, 2], [-1, 0, 1]],
    ])
    actual = training_ce_loss(logits, targets, ignore_label=-1)
    resized = F.interpolate(
        logits, targets.shape[-2:], mode="bilinear", align_corners=True
    )
    expected = F.cross_entropy(resized, targets, ignore_index=-1)
    actual_gradient = torch.autograd.grad(
        actual, logits, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        actual_gradient, expected_gradient, atol=1e-7, rtol=1e-7
    )
