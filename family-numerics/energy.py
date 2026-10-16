"""
Multiplicative-energy prediction for the 4th moment of the family rows.

If the rows u average chi_n(u) like independent random sextic values per prime
(exact orthogonality), then
    E_u |A_u|^4 = sum_m ( sum_{n1 n2 = m} c(n1) c(n2) )^2,
    E_u |A_u|^2 = sum_n c(n)^2,
with c(n) = mu(n) W(N n / D) on squarefree ideals n coprime to 6, and
n1 n2 = m as ideals. For squarefree solutions mu(n1)mu(n2)mu(n3)mu(n4) = +1, so
the off-diagonal solutions add to the Gaussian value 2 (sum c^2)^2.

Ideals are hashed additively (random 64-bit value per prime ideal), so the hash
of n1 n2 is h(n1) + h(n2) mod 2^64. Prints the predicted ratio and the measured
raw ratio E|A|^4 / (E|A|^2)^2 over all rows of the matching run.

  python energy.py D [D ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, squarefree_ideals
from family import W


def predicted_ratio(D, seed=7):
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    c = (mu * W(norms / D)).astype(np.float64)
    keep = c != 0
    rnd = np.random.default_rng(seed).integers(0, 2**63, len(P["norm"]), dtype=np.uint64)
    h = np.array([rnd[idx[ptr[i]:ptr[i + 1]]].sum(dtype=np.uint64) for i in range(len(c))],
                 dtype=np.uint64)[keep]
    c = c[keep]
    s2 = np.sum(c**2)
    # all ordered pairs (n1, n2): hash and weight, then group by hash
    H = (h[:, None] + h[None, :]).ravel()
    Wt = (c[:, None] * c[None, :]).ravel()
    order = np.argsort(H, kind="stable")
    H, Wt = H[order], Wt[order]
    starts = np.r_[0, np.nonzero(np.diff(H))[0] + 1]
    g = np.add.reduceat(Wt, starts)
    return np.sum(g**2) / s2**2, len(c)


def measured_ratio(D):
    d = np.load(f"runs/D{D}.npz")
    a2 = np.abs(d["A"]) ** 2
    return np.mean(a2**2) / np.mean(a2) ** 2, len(a2)


if __name__ == "__main__":
    print(f"{'D':>7} {'columns':>8} | {'predicted':>9} | {'measured (all rows)':>19}")
    for D in map(int, sys.argv[1:]):
        p, nc = predicted_ratio(D)
        m, nr = measured_ratio(D)
        print(f"{D:7d} {nc:8d} | {p:9.4f} | {m:9.4f} ({nr} rows)", flush=True)
