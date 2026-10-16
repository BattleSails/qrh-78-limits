"""
Mean square of the sextic family in the quasi-Riemann argument (OpenAI, Oct 2026, paper2 Sec. 2-3).

  A_u(D) = sum_n mu(n) chi_n(u) W(N n / D),   chi_n(u) = (u/n)_6,

over squarefree ideals n coprime to 6 (nu = trivial character), with W a smooth
bump supported in [1, 2]. For every nonzero u in Z[w] with N(u) <= Hmax we store
A_u(D) and its coefficient mass sum_{(n,u)=1} |mu W|^2 (the random-model
expectation of |A_u|^2); the mean square over any H <= Hmax is then a cumulative sum.

  python family.py D KAPPA_MAX OUT.npz
"""
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals, elements_upto


def W(x):
    """Smooth bump on (1, 2), max 1 at x = 1.5."""
    t = 2 * x - 3
    out = np.zeros_like(x, dtype=np.float64)
    m = np.abs(t) < 1
    out[m] = np.exp(1 - 1 / (1 - t[m] ** 2))
    return out


@njit(parallel=True, cache=True)
def family_sums(ux, uy, kind, p, r, pw, coef, ptr, idx):
    nU, nP, nN = len(ux), len(p), len(coef)
    re6 = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    im6 = np.array([0.0, 0.8660254037844386, 0.8660254037844386, 0.0,
                    -0.8660254037844386, -0.8660254037844386])
    Are = np.zeros(nU)
    Aim = np.zeros(nU)
    mass = np.zeros(nU)
    for k in prange(nU):
        e = np.empty(nP, dtype=np.int64)
        for i in range(nP):
            e[i] = sextic_exponent(ux[k], uy[k], kind[i], p[i], r[i], pw[i])
        sr = 0.0
        si = 0.0
        sm = 0.0
        for n in range(nN):
            s = 0
            zero = False
            for t in range(ptr[n], ptr[n + 1]):
                v = e[idx[t]]
                if v < 0:
                    zero = True
                    break
                s += v
            if not zero:
                s %= 6
                sr += coef[n] * re6[s]
                si += coef[n] * im6[s]
                sm += coef[n] * coef[n]
        Are[k] = sr
        Aim[k] = si
        mass[k] = sm
    return Are, Aim, mass


def run(D, kappa_max, out):
    t0 = time.time()
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    coef = mu * W(norms / D)
    keep = coef != 0
    # rebuild CSR for the kept columns
    new_ptr = np.zeros(keep.sum() + 1, dtype=np.int64)
    lens = np.diff(ptr)[keep]
    new_ptr[1:] = np.cumsum(lens)
    new_idx = np.concatenate([idx[ptr[i]:ptr[i + 1]] for i in np.nonzero(keep)[0]])
    coef = coef[keep].astype(np.float64)
    Hmax = int(D ** kappa_max)
    ux, uy, un = elements_upto(Hmax)
    print(f"D={D}: {len(P['norm'])} prime ideals, {len(coef)} columns, {len(ux)} rows (N(u) <= {Hmax:,})",
          flush=True)
    Are, Aim, mass = family_sums(ux, uy, P["kind"], P["p"], P["r"], P["pw"], coef, new_ptr, new_idx)
    np.savez(out, D=D, ux=ux, uy=uy, un=un, A=Are + 1j * Aim, mass=mass, coef=coef,
             col_norm=norms[keep], kappa_max=kappa_max)
    print(f"  done in {time.time() - t0:.1f} s -> {out}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
