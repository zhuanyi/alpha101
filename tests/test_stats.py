import numpy as np
import pandas as pd
import pytest

from alpha101.stats import adjusted_threshold, effective_number_of_tests, newey_west_t, rank_ic


def test_rank_ic_perfect_and_inverse():
    idx = pd.bdate_range("2024-01-01", periods=3)
    a = pd.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, np.nan, 4]], index=idx, dtype=float)
    f = pd.DataFrame([[10, 20, 30, 40], [10, 20, 30, 40], [1, 2, 3, 4]], index=idx, dtype=float)
    out = rank_ic(a, f, min_names=3)
    np.testing.assert_allclose(out["ic"], [1, -1, 1])
    np.testing.assert_array_equal(out["n"], [4, 4, 3])
    assert np.isnan(rank_ic(a, f, min_names=4)["ic"].iloc[2])


def test_effective_number_of_tests():
    assert effective_number_of_tests(np.eye(4)) == pytest.approx(4)
    assert effective_number_of_tests(np.ones((4, 4))) == pytest.approx(1)


def test_adjusted_threshold_monotone():
    assert adjusted_threshold(1) == pytest.approx(1.959964, rel=1e-5)
    assert adjusted_threshold(10) > adjusted_threshold(2) > adjusted_threshold(1)


def test_newey_west_iid_close_to_plain_t():
    x = np.random.default_rng(0).normal(0.1, 1, 5000)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, 0) == pytest.approx(plain, rel=1e-9)
    assert newey_west_t(x, 5) == pytest.approx(plain, rel=0.1)
