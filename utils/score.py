import numpy as np

SCALE_FACTOR = 1000
SCORE_FLOOR = -10 ** 6


def per_sample_score(true, mean, errorbar):
    """Phase 1 challenge score for each map (higher is better)."""
    sq = (true - mean) ** 2
    return -np.sum(sq / errorbar ** 2 + np.log(errorbar ** 2) + SCALE_FACTOR * sq, axis=1)


def score_phase1(true, mean, errorbar):
    """Mean Phase 1 score, floored at -1e6 as on Codabench."""
    return max(float(np.mean(per_sample_score(true, mean, errorbar))), SCORE_FLOOR)
