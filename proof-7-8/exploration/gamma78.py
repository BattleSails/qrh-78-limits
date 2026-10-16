"""Break-even signal sensitivity for the conservative model (model78c).
Replace the signal h/6 in C(sigma0) = sigma0 + l_x/2 - 1 + h/6 by gamma*h.
  variant 'all' : C(sigma0) changes everywhere it is subtracted (low, high_h, high_mid, floor,
                  small, mid_t1); this is the consistent reading of paper.tex:6151.
  variant 'low' : only the low estimate changes (literal reading; E-constraints keep 1/6).
All noise and count inputs are unchanged, including c = 1/(6 kappa) and the clamp kappa >= 3/4.
Usage: python gamma78.py GAMMA VARIANT K0
"""
import sys, numpy as np
from scipy.optimize import differential_evolution, minimize
from model78c import check as check_c
SHIFTED = ('low', 'high_h', 'high_mid', 'floor', 'small', 'mid_t1')
def check(g, gam, var, k0=1.0, nd=14, nx=4):
    out = check_c(g, k0, nd, nx)
    h = 1 - g['lx'] + g['ell']; dc = (gam - 1/6)*h
    for k in (SHIFTED if var == 'all' else ('low',)):
        out[k] = float(out[k]) - dc
    return out
def worst(p, s0, gam, var, k0, nd=14, nx=4):
    if p[0] < 0: return 1 + abs(p[0])
    return max(check(dict(b=p[0], lx=p[1], ell=p[2], s0=s0), gam, var, k0, nd, nx).values())
BOUNDS = [(0.0, 0.4), (0.05, 0.7), (0.0, 0.5)]
def feasible(s0, gam, var, k0, p):
    r = differential_evolution(worst, BOUNDS, args=(s0, gam, var, k0), seed=1, maxiter=30,
                               popsize=8, tol=1e-8, polish=False, x0=p)
    r2 = minimize(worst, r.x, args=(s0, gam, var, k0, 30, 8), method='Nelder-Mead',
                  options=dict(xatol=1e-6, fatol=1e-9, maxiter=300))
    return (r2.fun, r2.x) if r2.fun < r.fun else (r.fun, r.x)
if __name__ == "__main__":
    gam, var, k0 = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    lo, hi = 0.60, 0.8755; p = np.array([0.0759, 0.3731, 0.1773]); best = None
    for it in range(12):
        mid = (lo + hi)/2; f, x = feasible(mid, gam, var, k0, p)
        if f < -1e-6: hi, p, best = mid, x, x
        else: lo = mid
        print(f"g={gam:.4f} {var} s0={mid:.5f} worst={f:+.2e} p={np.round(x,4)}", flush=True)
    if best is None:
        print("RESULT", gam, var, k0, "none"); sys.exit()
    v = check(dict(b=p[0], lx=p[1], ell=p[2], s0=hi), gam, var, k0, nd=80, nx=11)
    bind = [k for k, a in v.items() if a > -2e-3]
    print("RESULT", gam, var, k0, round(hi, 5), np.round(p, 5), "binding:", bind,
          {k: round(float(a), 5) for k, a in v.items()}, flush=True)
