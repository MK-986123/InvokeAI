import pytest

from invokeai.app.invocations.resample_banding_fix import _compute_window


def test_compute_window_full_strength():
    start, frac = _compute_window(1.0, 14)
    assert start == 0
    assert pytest.approx(frac, rel=1e-6) == 0.0


def test_compute_window_half_strength():
    start, frac = _compute_window(0.5, 14)
    assert start == 7
    assert pytest.approx(frac, rel=1e-6) == 0.5


def test_compute_window_strength_clamped():
    start, frac = _compute_window(1.5, 10)
    assert start == 0
    assert pytest.approx(frac, rel=1e-6) == 0.0


def test_compute_window_zero_strength_returns_last_step():
    start, frac = _compute_window(0.0, 10)
    assert start == 9
    assert pytest.approx(frac, rel=1e-6) == 0.9


@pytest.mark.parametrize("steps", [0, -1])
def test_compute_window_invalid_steps(steps: int):
    with pytest.raises(ValueError):
        _compute_window(0.5, steps)
