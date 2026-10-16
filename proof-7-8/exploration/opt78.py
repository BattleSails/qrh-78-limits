import sys, numpy as np
from scipy.optimize import differential_evolution
from model78 import check
def worst(p, s0, k0):
    g = dict(b=p[0], lx=p[1], ell=p[2], s0=s0)
    return max(check(g, k0, nd=14, nx=4).values())
def feasible(s0, k0):
    r = differential_evolution(worst, [(0.0, 0.35), (0.1, 0.55), (0.02, 0.3)],
                               args=(s0, k0), seed=1, maxiter=40, popsize=10, tol=1e-7, polish=True)
    return r.fun, r.x
k0 = float(sys.argv[1])
lo, hi = 0.86, 0.88
for it in range(10):
    mid = (lo+hi)/2
    f, x = feasible(mid, k0)
    if f < -1e-6: hi, best = mid, (f, x)
    else: lo = mid
    print(f"k0={k0} s0={mid:.5f} worst={f:+.2e} b,lx,ell={np.round(x,4)}", flush=True)
print("RESULT", k0, hi, best)
