"""
Return probability of the simple random walk on Z^d (Polya's theorem).

For each dimension d we start N independent walkers at the origin.
At every step each walker picks one of its 2d neighbours uniformly at random.
We record the first time each walker is back at the origin, and report
u_hat(T) = (number of walkers that returned by step T) / N
at several checkpoints T.

Walkers that have already returned are removed ("dead") so the loop gets cheaper.
"""
import sys
import json
import numpy as np

N = 10_000
CHECKPOINTS = [10**2, 10**3, 10**4, 10**5]
T_MAX = CHECKPOINTS[-1]


def simulate(d, seed):
    rng = np.random.default_rng(seed)
    pos = np.zeros((N, d), dtype=np.int32)      # positions of walkers still "alive"
    alive_ids = np.arange(N)                     # which original walker each row is
    returned_by = {}                             # checkpoint -> count returned
    n_returned = 0

    for t in range(1, T_MAX + 1):
        m = pos.shape[0]
        if m == 0:
            # everyone has returned; fill remaining checkpoints
            for c in CHECKPOINTS:
                returned_by.setdefault(c, n_returned)
            break
        axis = rng.integers(0, d, size=m)        # which coordinate moves
        sign = rng.integers(0, 2, size=m) * 2 - 1  # +1 or -1
        pos[np.arange(m), axis] += sign
        back = ~pos.any(axis=1)                  # True where all coordinates are 0
        k = int(back.sum())
        if k:
            n_returned += k
            keep = ~back
            pos = pos[keep]
        if t in CHECKPOINTS:
            returned_by[t] = n_returned

    return {c: returned_by[c] / N for c in CHECKPOINTS}


if __name__ == "__main__":
    d = int(sys.argv[1])
    res = simulate(d, seed=1000 + d)
    out = {"d": d, "N": N, "u_hat": {str(k): v for k, v in res.items()},
           "se": {str(k): float(np.sqrt(v * (1 - v) / N)) for k, v in res.items()}}
    print(json.dumps(out))
