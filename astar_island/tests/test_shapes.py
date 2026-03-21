"""Shape tests: verify that infer_full_map returns [seeds, H, W, 6]."""
import numpy as np
import pytest
from astar_island.online.round_state import RoundState
from astar_island.online.infer import infer_full_map
from astar_island.validators import validate_prediction_tensor


def _make_state(H=40, W=40, seeds=5):
    grid = [[0] * W for _ in range(H)]
    # Add some variety
    for i in range(5):
        grid[i][i] = 10   # ocean
        grid[i+5][i] = 5  # mountain
        grid[i+10][i] = 4  # forest
        grid[i+15][i] = 1  # settlement
    return RoundState.from_detail({
        "id": "shape-test",
        "map_width": W,
        "map_height": H,
        "seeds_count": seeds,
        "status": "active",
        "initial_states": [{"grid": grid, "settlements": []} for _ in range(seeds)],
    })


class TestInferFullMapShapes:
    def test_output_shape(self):
        state = _make_state(H=40, W=40, seeds=5)
        pred = infer_full_map(state)
        assert pred.shape == (5, 40, 40, 6), f"Got {pred.shape}"

    def test_output_shape_non_square(self):
        state = _make_state(H=20, W=30, seeds=3)
        pred = infer_full_map(state)
        assert pred.shape == (3, 20, 30, 6)

    def test_each_seed_tensor_valid(self):
        state = _make_state(H=40, W=40, seeds=5)
        pred = infer_full_map(state)
        for s in range(5):
            validate_prediction_tensor(pred[s], 40, 40)

    def test_floor_applied(self):
        state = _make_state()
        pred = infer_full_map(state, prob_floor=0.005)
        assert np.all(pred >= 0.005), "floor_renorm should ensure min >= 0.005"

    def test_sums_to_1(self):
        state = _make_state()
        pred = infer_full_map(state)
        sums = pred.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-4)

    def test_static_ocean_cells_have_ocean_prior(self):
        state = _make_state()
        pred = infer_full_map(state)
        # Cells where raw_grids[0] == 10 should have high class-0 probability
        for s in range(5):
            ocean_rows, ocean_cols = np.where(state.raw_grids[s] == 10)
            for r, c in zip(ocean_rows, ocean_cols):
                assert pred[s, r, c, 0] > 0.5, \
                    f"Ocean cell ({r},{c}) seed {s}: class 0 prob = {pred[s,r,c,0]:.3f}"

    def test_static_mountain_cells_have_mountain_prior(self):
        state = _make_state()
        pred = infer_full_map(state)
        for s in range(5):
            mtn_rows, mtn_cols = np.where(state.raw_grids[s] == 5)
            for r, c in zip(mtn_rows, mtn_cols):
                assert pred[s, r, c, 5] > 0.5, \
                    f"Mountain cell ({r},{c}) seed {s}: class 5 prob = {pred[s,r,c,5]:.3f}"
