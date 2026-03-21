import numpy as np
import pytest
from astar_island.map.encoding import (
    encode_grid, is_raw_static, class_prior, RAW_TO_CLASS, STATIC_RAW_CODES
)


def test_raw_10_encodes_to_class_0():
    grid = [[10, 11, 0]]
    enc = encode_grid(grid)
    assert enc[0, 0] == 0, "ocean (raw 10) → class 0"
    assert enc[0, 1] == 0, "plains (raw 11) → class 0"
    assert enc[0, 2] == 0, "empty (raw 0) → class 0"


def test_raw_5_encodes_to_class_5():
    grid = [[5]]
    enc = encode_grid(grid)
    assert enc[0, 0] == 5, "mountain (raw 5) → class 5"


def test_all_known_raw_codes():
    for raw, expected_class in RAW_TO_CLASS.items():
        grid = [[raw]]
        enc = encode_grid(grid)
        assert enc[0, 0] == expected_class, f"raw {raw} → class {expected_class}"


def test_is_raw_static_ocean():
    assert is_raw_static(10), "ocean (raw 10) is static"


def test_is_raw_static_mountain():
    assert is_raw_static(5), "mountain (raw 5) is static"


def test_is_raw_static_non_static():
    for code in [0, 1, 2, 3, 4, 11]:
        assert not is_raw_static(code), f"raw {code} should NOT be static"


def test_class_prior_sums_to_1():
    for raw_code in [0, 1, 2, 3, 4, 5, 10, 11]:
        prior = class_prior(raw_code)
        assert prior.shape == (6,)
        assert abs(prior.sum() - 1.0) < 1e-6, f"Prior for raw {raw_code} does not sum to 1"


def test_class_prior_no_zeros():
    for raw_code in [0, 1, 2, 3, 4, 5, 10, 11]:
        prior = class_prior(raw_code)
        assert np.all(prior > 0), f"Prior for raw {raw_code} has zeros"


def test_static_raw_codes_set():
    assert 5 in STATIC_RAW_CODES
    assert 10 in STATIC_RAW_CODES
    assert 0 not in STATIC_RAW_CODES  # class 0 is NOT synonymous with static
