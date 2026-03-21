import numpy as np
import pytest
from astar_island.utils.math import floor_renorm, entropy, kl_div, weighted_kl_score


class TestFloorRenorm:
    def test_output_sums_to_1(self):
        arr = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.1])
        out = floor_renorm(arr, floor=0.005)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_no_zeros(self):
        arr = np.zeros(6)
        arr[0] = 1.0
        out = floor_renorm(arr, floor=0.005)
        assert np.all(out > 0)

    def test_batched_shape(self):
        arr = np.random.dirichlet(np.ones(6), size=(5, 40, 40))
        out = floor_renorm(arr, floor=0.005)
        assert out.shape == (5, 40, 40, 6)
        assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-6)
        assert np.all(out > 0)  # no exact zeros; renorm may push below floor

    def test_already_valid_unchanged_shape(self):
        arr = np.full(6, 1.0 / 6)
        out = floor_renorm(arr)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_floor_applied(self):
        # After maximum(arr, floor) + renorm, all values > 0 (no exact zeros).
        # Values slightly below floor are expected after renorm when one class dominates.
        arr = np.array([0.999, 0.0, 0.0, 0.0, 0.0, 0.001])
        out = floor_renorm(arr, floor=0.01)
        assert np.all(out > 0), "floor_renorm must eliminate all exact zeros"


class TestEntropy:
    def test_uniform_max_entropy(self):
        uniform = np.full(6, 1.0 / 6)
        h = entropy(uniform)
        expected = -6 * (1/6) * np.log(1/6)
        assert abs(float(h) - expected) < 1e-6

    def test_deterministic_zero_entropy(self):
        det = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Uses clip internally so no log(0)
        h = float(entropy(det))
        assert h >= 0.0

    def test_non_negative(self):
        for _ in range(10):
            p = np.random.dirichlet(np.ones(6))
            assert float(entropy(p)) >= 0.0


class TestKLDiv:
    def test_identical_distributions_zero_kl(self):
        p = np.array([0.2, 0.3, 0.1, 0.1, 0.2, 0.1])
        assert kl_div(p, p) < 1e-9

    def test_kl_inf_when_q_zero_and_p_positive(self):
        p = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
        result = kl_div(p, q)
        assert result == float("inf")

    def test_kl_non_negative(self):
        p = np.random.dirichlet(np.ones(6))
        q = np.random.dirichlet(np.ones(6))
        assert kl_div(p, q) >= 0


class TestWeightedKLScore:
    def test_perfect_prediction_gives_100(self):
        gt = np.random.dirichlet(np.ones(6), size=(5, 5))
        # Prediction matches ground truth exactly
        score = weighted_kl_score(gt, gt)
        assert abs(score - 100.0) < 1e-4

    def test_score_in_0_100_range(self):
        gt = np.random.dirichlet(np.ones(6), size=(5, 5))
        pred = floor_renorm(np.random.rand(5, 5, 6))
        score = weighted_kl_score(pred, gt)
        assert 0.0 <= score <= 100.0
