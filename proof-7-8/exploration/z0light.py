"""Light test: cubic signal gamma = 1/3 with the Euler-region contour z0 as an input.
Coarse grid (nd=14, nx=4), short DE, 8 bisection steps on s0 in [0.6, 0.99]. Variant 'all'.
Usage: python z0light.py Z0 [GAMMA]"""
import sys, numpy as np
import model78
model78.z0 = float(sys.argv[1])
import gamma78
gam = float(sys.argv[2]) if len(sys.argv) > 2 else 1/3
from scipy.optimize import differential_evolution
def worst(p, s0): return gamma78.worst(p, s0, gam, 'all', 1.0)
lo, hi, best = 0.60, 0.99, None
for it in range(8):
    mid = (lo+hi)/2
    r = differential_evolution(worst, gamma78.BOUNDS, args=(mid,), seed=1, maxiter=20, popsize=8, tol=1e-8)
    if r.fun < -1e-6: hi, best = mid, r.x
    else: lo = mid
if best is None: print("RESULT z0", sys.argv[1], "gamma", gam, "infeasible below 0.99"); sys.exit()
v = gamma78.check(dict(b=best[0], lx=best[1], ell=best[2], s0=hi), gam, 'all')
print("RESULT z0", sys.argv[1], "gamma", round(gam,4), "s0<=", round(hi,4), np.round(best,3),
      "binding:", [k for k, a in v.items() if a > -3e-3], flush=True)
