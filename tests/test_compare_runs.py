"""Unit tests for script/compare_runs.py (stdlib only)."""

import importlib.util
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "compare_runs", os.path.join(_REPO_ROOT, "script", "compare_runs.py")
)
compare_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_runs)


def _run(scores, bucket="b"):
    return {f"{i:016x}": {"bucket": bucket, "score": s} for i, s in enumerate(scores)}


def test_identical_runs_have_zero_delta():
    base = _run([0.5, 0.8, 1.0, 0.0])
    r = compare_runs.compare(base, base, n_boot=200)
    assert r["delta_mean"] == 0.0
    assert r["ci"] == (0.0, 0.0)
    assert (r["wins"], r["ties"], r["losses"]) == (0, 4, 0)
    assert r["sign_p"] == 1.0


def test_single_recovered_scenario_is_not_significant():
    """One 0 -> 0.65 recovery out of 84 is the mini_eval_v2 re-ranker pattern."""
    base_scores = [0.9] * 83 + [0.0]
    cand_scores = [0.9] * 83 + [0.65]
    r = compare_runs.compare(_run(base_scores), _run(cand_scores), n_boot=2000)
    assert r["delta_mean"] == pytest.approx(0.65 / 84)
    assert r["ci"][0] == 0.0          # CI touches zero: one scenario drives it
    assert r["sign_p"] == 1.0         # 1 win, 0 losses
    assert r["delta_without_top"] == pytest.approx(0.0)


def test_consistent_improvement_is_significant():
    base = _run([0.5] * 30)
    cand = _run([0.6] * 30)
    r = compare_runs.compare(base, cand, n_boot=500)
    assert r["wins"] == 30
    assert r["ci"][0] > 0
    assert r["sign_p"] < 1e-6


def test_only_shared_tokens_are_compared():
    base = _run([0.5, 0.5, 0.5])
    cand = {k: v for k, v in list(_run([0.7, 0.7, 0.7]).items())[:2]}
    r = compare_runs.compare(base, cand, n_boot=100)
    assert r["n"] == 2
    assert len(r["only_base"]) == 1


def test_sign_test_matches_binomial():
    # 9 wins, 1 loss: two-sided p = 2 * (C(10,0) + C(10,1)) / 2^10
    assert compare_runs.sign_test_p(9, 1) == pytest.approx(22 / 1024)


def test_split_is_deterministic_and_roughly_balanced():
    tokens = [f"{i:016x}" for i in range(400)]
    first = [compare_runs.split_of(t) for t in tokens]
    assert first == [compare_runs.split_of(t) for t in tokens]
    assert 150 < first.count("tune") < 250
