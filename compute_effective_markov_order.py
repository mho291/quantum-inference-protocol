import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

import numpy as np


def compute_trace_distance(p1, p2):
    """Compute the trace distance between two discrete probability distributions.
    The input vectors are internally normalized before computing the trace
    distance.

    Args:
        p1 (np.ndarray): First probability distribution.
        p2 (np.ndarray): Second probability distribution.
    Returns:
        trace distance (float): Trace distance between p1 and p2,
            defined as 0.5 * sum(abs(p1 - p2)).
    """
    if np.sum(p1) == 0 or np.sum(p2) == 0:
        return np.nan

    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    return 0.5 * np.sum(np.abs(p1 - p2))


def distances_from_Lmin_Lmax(bits, len_min, len_max):
    """Compute trace distances between conditional distributions over a range of
    history lengths.

    For each history length L, the function computes the trace distance
    between coonditional probability distributions
        P(X_0 | 0 X_{-L:0})
        P(X_0 | 1 X_{-L:0})
    where the (L+1)-th past is either 0 or 1.

    Args:
        bits (np.ndarray): Binary stochastic process.
        len_min (int): Minimum history length L.
        len_max (int): Maximum history length L.
    Returns:
        trace_distance_all_L (np.ndarray): Trace distances for all L
            from len_min to len_max.
    """

    trace_distance_all_L = np.zeros(len_max - len_min + 1)
    for idx, L in enumerate(range(len_min, len_max + 1)):

        windows = np.lib.stride_tricks.sliding_window_view(bits, window_shape=L + 2)
        sub = windows[:, :-1]
        weights = 2 ** np.arange(sub.shape[1] - 1, -1, -1)
        ints = sub @ weights

        condCounts = np.zeros((2 ** (L + 1), 2))
        for i in range(len(windows)):
            r_idx = ints[i]
            c_idx = windows[i, -1]
            condCounts[r_idx, c_idx] += 1
        row_sums = condCounts.sum(axis=1, keepdims=True)
        condProb = np.zeros_like(condCounts)
        np.divide(condCounts, row_sums, out=condProb, where=row_sums != 0)
        half = condProb.shape[0] // 2
        P2 = condProb[:half]
        P3 = condProb[half:]

        TD_vector = np.zeros(2**L)
        v = 0
        for i in range(2**L):
            TD_vector[i] = compute_trace_distance(P2[i], P3[i])
            if np.isnan(TD_vector[i]):
                v += 1
                TD_vector[i] = 0
        trace_distance = np.sum(TD_vector) / (2**L - v)
        trace_distance_all_L[idx] = trace_distance
        logging.info(
            f"[{idx+1}/{len_max-len_min+1}] L={L}, trace distance={trace_distance:.6f}"
        )

    return trace_distance_all_L


def find_effective_markov_order(bits, len_min, len_max, delta=None):
    """Estimate the effective Markov order of a binary stochastic process.
    The effective Markov order is defined as the smallest history length L+1
    such that the trace distances for L+1 up to L+4 are all below
    a threshold delta. The L+1 to L+4 condition reduces false positives
    caused by finite-sample fluctuations in empirically estimated conditional
    probabilities at large L.

    If delta is not provided, the default threshold is:
        delta = 1 / sqrt(len(bits))

    Args:
        bits (np.ndarray): Binary stochastic process.
        len_min (int): Minimum history length L to test.
        len_max (int): Maximum history length L to test.
        delta (float, optional): Trace distance threshold.

    Returns:
        tuple:
            - L (int or None): Estimated effective Markov order.
                Returns None if no valid L satisfies the condition.
            - trace_distances (np.ndarray): Trace distances for all tested L.
            - delta (float): Threshold used in the computation.
    """

    if delta is None:
        delta = 1 / np.sqrt(len(bits))
    trace_distances = distances_from_Lmin_Lmax(bits, len_min, len_max)
    for idx in range(len(trace_distances) - 3):
        L = len_min + idx
        if np.all(trace_distances[idx : idx + 4] <= delta):
            logging.info(f"Effective Markov order {L}, delta = {delta:.12f}")
            return L, trace_distances, delta
    logging.info(f"Effective Markov order None, delta = {delta:.12f}")
    return None, trace_distances, delta
