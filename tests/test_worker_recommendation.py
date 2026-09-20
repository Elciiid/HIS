"""How many generation workers the measurement should recommend.

The rungs used here are the ones Kaggle actually measured (2 x Tesla T4, 4 vCPUs, 4 storms per
rung): 6.77, 12.91 and 13.42 storms/hour at 1, 2 and 4 workers, with peak host RSS of 3.84,
6.84 and 13.61 GB. Four workers is the fastest and the wrong answer -- it is 3.8% quicker than
two for twice the host memory, on a machine whose RAM limit is tighter than its VRAM."""
import pytest

from hydrointel.kaggle_support import KNEE_TOL, recommend_workers

MEASURED = [{"workers": 1, "storms_per_hour": 6.77, "peak_rss_gb": 3.84, "ok": True},
            {"workers": 2, "storms_per_hour": 12.91, "peak_rss_gb": 6.84, "ok": True},
            {"workers": 4, "storms_per_hour": 13.42, "peak_rss_gb": 13.61, "ok": True}]


def test_picks_the_knee_not_the_peak():
    n, why = recommend_workers(MEASURED, identical=True)
    assert n == 2
    assert "4 workers" in why and "13.42" in why          # names what it passed up, and by how much
    assert "0.50x its host RAM" in why


def test_peak_is_taken_when_it_is_a_real_gain():
    rows = [dict(r) for r in MEASURED]
    rows[2]["storms_per_hour"] = 25.0                    # 4 workers genuinely twice as fast as 2
    n, why = recommend_workers(rows, identical=True)
    assert n == 4
    assert why == "fastest measured and bit-identical to the serial run"


def test_a_difference_from_the_serial_run_vetoes_parallelism():
    n, why = recommend_workers(MEASURED, identical=False)
    assert n == 1
    assert "NOT bit-identical" in why


def test_no_successful_rung_falls_back_to_one():
    n, why = recommend_workers([], identical=True)
    assert n == 1
    assert why == "no run succeeded"


@pytest.mark.parametrize("ratio, expect", [(KNEE_TOL + 0.01, 2), (KNEE_TOL - 0.01, 4)])
def test_the_tolerance_is_the_boundary(ratio, expect):
    """A rung counts as 'as fast as the best' within KNEE_TOL, and the boundary is not fudged."""
    rows = [dict(r) for r in MEASURED]
    rows[1]["storms_per_hour"] = ratio * rows[2]["storms_per_hour"]
    assert recommend_workers(rows, identical=True)[0] == expect
