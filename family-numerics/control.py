"""
Control runs for the family experiment: same columns and rows as family.py,
but with the Mobius coefficients replaced.

  mode "random": mu(n) -> independent random signs (fixed seed)
  mode "plain" : mu(n) -> 1  (the sum without Mobius)

  python control.py D KAPPA_MAX MODE OUT.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, squarefree_ideals, elements_upto
from family import W, family_sums


def run(D, kappa_max, mode, out):
    t0 = time.time()
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    w = W(norms / D)
    keep = w != 0
    if mode == "random":
        sign = np.random.default_rng(12345).choice([-1.0, 1.0], size=len(norms))
    elif mode == "plain":
        sign = np.ones(len(norms))
    else:
        raise ValueError(mode)
    coef = (sign * w)[keep]
    new_ptr = np.zeros(keep.sum() + 1, dtype=np.int64)
    new_ptr[1:] = np.cumsum(np.diff(ptr)[keep])
    new_idx = np.concatenate([idx[ptr[i]:ptr[i + 1]] for i in np.nonzero(keep)[0]])
    ux, uy, un = elements_upto(int(D ** kappa_max))
    Are, Aim, mass = family_sums(ux, uy, P["kind"], P["p"], P["r"], P["pw"], coef, new_ptr, new_idx)
    np.savez(out, D=D, ux=ux, uy=uy, un=un, A=Are + 1j * Aim, mass=mass, coef=coef, mode=mode)
    print(f"D={D} mode={mode}: {len(ux)} rows, {time.time() - t0:.1f} s -> {out}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4])
