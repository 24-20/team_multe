"""Test RoundState: construction, coordinate conversion, posterior updates."""
import numpy as np
import pytest
from astar_island.online.round_state import RoundState


def _make_detail(H=10, W=10, seeds=2):
    grid = [[0] * W for _ in range(H)]
    grid[0][0] = 10  # ocean top-left
    grid[H-1][W-1] = 5  # mountain bottom-right
    return {
        "id": "test-round",
        "map_width": W,
        "map_height": H,
        "seeds_count": seeds,
        "status": "active",
        "initial_states": [
            {"grid": grid, "settlements": [{"x": 3, "y": 3, "has_port": False, "alive": True}]},
            {"grid": grid, "settlements": []},
        ][:seeds],
    }


class TestRoundStateConstruction:
    def test_shape(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        assert state.raw_grids.shape == (2, 10, 10)
        assert state.encoded_grids.shape == (2, 10, 10)
        assert state.prior_probs.shape == (2, 10, 10, 6)
        assert state.empirical_counts.shape == (2, 10, 10, 6)

    def test_static_cell_ocean(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        # raw_grids[0, 0, 0] == 10 (ocean)
        assert state.raw_grids[0, 0, 0] == 10

    def test_prior_sums_to_1(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        sums = state.prior_probs.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_posterior_mean_equals_prior_before_queries(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        posterior = state.posterior_mean
        prior = state.prior_probs
        # With no observations, posterior ≈ prior (they match exactly by design)
        assert np.allclose(posterior, prior, atol=1e-6)


class TestCoordinateConversion:
    def test_update_from_query_uses_xy_to_rowcol(self):
        """API gives (x=col, y=row); internally (row, col) = (y, x)."""
        detail = _make_detail(H=10, W=10)
        state = RoundState.from_detail(detail)

        # Query viewport: x=2 (col 2), y=3 (row 3), w=3, h=2
        # Should update rows 3..4, cols 2..4
        grid_result = [[1, 1, 1], [1, 1, 1]]  # all settlement (class 1)
        state.update_from_query(
            seed_idx=0,
            viewport_x=2, viewport_y=3,
            viewport_w=3, viewport_h=2,
            grid_result=grid_result,
        )

        # Cells (row=3, col=2..4) and (row=4, col=2..4) should have settlement counts
        assert state.empirical_counts[0, 3, 2, 1] == 1.0
        assert state.empirical_counts[0, 4, 4, 1] == 1.0
        # Adjacent cells should NOT be updated
        assert state.empirical_counts[0, 2, 2, 1] == 0.0
        assert state.empirical_counts[0, 5, 2, 1] == 0.0

    def test_observed_count_increments(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        grid_result = [[4, 4], [4, 4]]  # forest
        state.update_from_query(
            seed_idx=0,
            viewport_x=1, viewport_y=1,
            viewport_w=2, viewport_h=2,
            grid_result=grid_result,
        )
        assert state.observed_count[0, 1, 1] == 1
        assert state.observed_count[0, 1, 2] == 1
        assert state.queries_used == 1


class TestPosteriorMean:
    def test_posterior_shifts_toward_observed_class(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)

        # Query a cell multiple times with same result (class 4 = forest)
        for _ in range(10):
            state.update_from_query(
                seed_idx=0,
                viewport_x=5, viewport_y=5,
                viewport_w=1, viewport_h=1,
                grid_result=[[4]],
            )

        # After 10 forest observations, forest class should dominate
        posterior = state.posterior_mean[0, 5, 5]
        assert posterior[4] > 0.7, f"Expected forest to dominate, got {posterior}"

    def test_posterior_sums_to_1(self):
        detail = _make_detail()
        state = RoundState.from_detail(detail)
        state.update_from_query(
            seed_idx=0, viewport_x=2, viewport_y=2,
            viewport_w=3, viewport_h=3,
            grid_result=[[1]*3]*3,
        )
        sums = state.posterior_mean.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-6)
