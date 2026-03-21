import numpy as np
import pytest
from astar_island.validators import (
    validate_prediction_tensor,
    validate_viewport,
    validate_round_detail,
)
from astar_island.utils.math import floor_renorm


def _valid_tensor(H=40, W=40):
    arr = np.random.dirichlet(np.ones(6), size=(H, W))
    return floor_renorm(arr.reshape(H, W, 6))


class TestValidatePredictionTensor:
    def test_valid_tensor_passes(self):
        t = _valid_tensor()
        validate_prediction_tensor(t, 40, 40)  # should not raise

    def test_wrong_shape_raises(self):
        t = _valid_tensor()
        with pytest.raises(ValueError, match="shape"):
            validate_prediction_tensor(t, 30, 40)

    def test_nan_raises(self):
        t = _valid_tensor()
        t[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_prediction_tensor(t, 40, 40)

    def test_inf_raises(self):
        t = _valid_tensor()
        t[0, 0, 1] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            validate_prediction_tensor(t, 40, 40)

    def test_negative_value_raises(self):
        t = _valid_tensor()
        t[1, 1, 2] = -0.001
        with pytest.raises(ValueError, match="negative"):
            validate_prediction_tensor(t, 40, 40)

    def test_does_not_sum_to_1_raises(self):
        t = np.full((40, 40, 6), 0.2)
        with pytest.raises(ValueError, match="sum"):
            validate_prediction_tensor(t, 40, 40)

    def test_exact_zero_raises(self):
        t = _valid_tensor()
        t[2, 3, 4] = 0.0
        t[2, 3] /= t[2, 3].sum()
        with pytest.raises(ValueError, match="zero"):
            validate_prediction_tensor(t, 40, 40)


class TestValidateViewport:
    def test_valid_viewport(self):
        validate_viewport(5, 5, 15, 15, 40, 40)

    def test_width_out_of_range(self):
        with pytest.raises(ValueError, match="width"):
            validate_viewport(0, 0, 16, 15, 40, 40)

    def test_height_out_of_range(self):
        with pytest.raises(ValueError, match="height"):
            validate_viewport(0, 0, 15, 0, 40, 40)

    def test_viewport_exceeds_map(self):
        with pytest.raises(ValueError):
            validate_viewport(30, 0, 15, 15, 40, 40)


class TestValidateRoundDetail:
    def _valid_detail(self):
        return {
            "id": "test-round",
            "map_width": 40,
            "map_height": 40,
            "seeds_count": 2,
            "status": "active",
            "initial_states": [
                {"grid": [[0]*40]*40, "settlements": []},
                {"grid": [[0]*40]*40, "settlements": []},
            ],
        }

    def test_valid_detail_passes(self):
        validate_round_detail(self._valid_detail())

    def test_missing_key_raises(self):
        d = self._valid_detail()
        del d["map_width"]
        with pytest.raises(ValueError, match="missing"):
            validate_round_detail(d)

    def test_wrong_seeds_count_raises(self):
        d = self._valid_detail()
        d["seeds_count"] = 5  # but only 2 initial_states
        with pytest.raises(ValueError):
            validate_round_detail(d)
