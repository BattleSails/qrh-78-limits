"""
Families of lower order: chi_n(u)^m with m = 2 (cubic symbol (u/n)_3) or
m = 3 (quadratic symbol (u/n)_2), with Mobius coefficients or (control) plain 1.

The exponent of (u/n)_6 is multiplied by m mod 6, so the validated sextic code
in eisenstein.py is reused unchanged. Copies are now u = v^3 (m = 2) or u = v^2
(m = 3).

  python family_order.py D KAPPA_MAX M MODE OUT.npz     (MODE = mobius | plain)
"""
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals, elements_upto
from family import W


@njit(parallel=True, cache=True)
def family_sums_pow(ux, uy, kind, p, r, pw, coef, ptr, idx, m):
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
                s = (s * m) % 6
                sr += coef[n] * re6[s]
                si += coef[n] * im6[s]
                sm += coef[n] * coef[n]
        Are[k] = sr
        Aim[k] = si
        mass[k] = sm
    return Are, Aim, mass


def run(D, kappa_max, m, mode, out):
    t0 = time.time()
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    w = W(norms / D)
    keep = w != 0
    coef = ((mu if mode == "mobius" else np.ones_like(mu)) * w)[keep].astype(np.float64)
    new_ptr = np.zeros(keep.sum() + 1, dtype=np.int64)
    new_ptr[1:] = np.cumsum(np.diff(ptr)[keep])
    new_idx = np.concatenate([idx[ptr[i]:ptr[i + 1]] for i in np.nonzero(keep)[0]])
    ux, uy, un = elements_upto(int(D ** kappa_max))
    Are, Aim, mass = family_sums_pow(ux, uy, P["kind"], P["p"], P["r"], P["pw"],
                                     coef, new_ptr, new_idx, m)
    np.savez(out, D=D, ux=ux, uy=uy, un=un, A=Are + 1j * Aim, mass=mass, m=m, mode=mode)
    print(f"D={D} m={m} {mode}: {len(ux)} rows, {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
