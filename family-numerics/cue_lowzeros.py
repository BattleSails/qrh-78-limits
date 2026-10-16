"""
CUE reference for lowzeros.py: the same two statistics (nearest-neighbour gap / its mean, and
|Lambda'(e^{i theta_j})| / its median) for Haar unitary matrices of size N, plus the
two-independent-factors model (union of two independent CUE spectra) for the factored rows.

  python cue_lowzeros.py
"""
import numpy as np
from scipy.stats import unitary_group

rng = np.random.default_rng(1)
EPS = np.array([0.05, 0.1, 0.2, 0.4])


def stats(th):
    th = np.sort(th % (2 * np.pi))
    N = len(th)
    g = np.diff(np.r_[th, th[0] + 2 * np.pi]) * N / (2 * np.pi)
    nn = np.minimum(g, np.roll(g, 1))
    z = np.exp(1j * th)
    dz = np.abs(z[:, None] - z[None, :]) + np.eye(N)
    lp = np.exp(np.log(dz).sum(1)) / N            # |Lambda'(z_j)| / N (density scale)
    return nn, lp


def run(N, reps, union):
    S, L = [], []
    for _ in range(reps):
        if union:
            th = np.r_[np.angle(np.linalg.eigvals(unitary_group.rvs(N // 2, random_state=rng))),
                       np.angle(np.linalg.eigvals(unitary_group.rvs(N // 2, random_state=rng)))]
        else:
            th = np.angle(np.linalg.eigvals(unitary_group.rvs(N, random_state=rng)))
        s, l = stats(th)
        S.append(s)
        L.append(l)
    return np.concatenate(S), np.concatenate(L)


for union, lab in ((False, "CUE"), (True, "two independent CUE")):
    for N in (20, 60):
        s, l = run(N, 120000 // N, union)
        out = []
        for v in (s / s.mean(), l / np.median(l)):
            F = np.array([np.mean(v < e) for e in EPS])
            sl = np.polyfit(np.log(EPS[F > 0]), np.log(F[F > 0]), 1)[0]
            out.append(" ".join(f"{f:.2e}" for f in F) + f"  slope {sl:.2f}")
        print(f"{lab:20s} N={N:3d}: mean nn gap {s.mean():.3f}\n   nn gap: {out[0]}\n   |L'|  : {out[1]}")
