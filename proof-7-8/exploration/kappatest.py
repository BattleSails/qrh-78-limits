"""Light test: count lemma with kappa = 2 s0 - 1 (no clamp at 3/4), all-cubic modelk.
At fixed s0, minimise the worst margin over (b, l_x, ell) twice: clamp kappa >= 3/4 and no clamp.
Usage: python kappatest.py K OFF S0"""
import sys, numpy as np
from scipy.optimize import differential_evolution
import modelk
s0 = float(sys.argv[3])
for kmin in (0.75, 0.0):
    modelk.KMIN = kmin
    r = differential_evolution(modelk.worst, [(0, 0.4), (0.05, 0.7), (0, 0.5)], args=(s0,), seed=1,
                               maxiter=25, popsize=8, tol=1e-8)
    v = modelk.check(*r.x, s0)
    print(f"RESULT k={sys.argv[1]} s0={s0} kappa={2*s0-1:.3f} clamp={kmin} worst={r.fun:+.4f} p={np.round(r.x,3)}",
          {n: round(float(a), 4) for n, a in v.items()}, flush=True)
