"""Naive order-k substitution in model78c (light test). Every sextic constant is replaced:
  signal h/6 -> h/k, low b/12 -> b/(2k), len2 11b/6 -> (2-1/k) b, e_long (1+5r)/6 -> (1+(k-1)r)/k,
  plain capacity c = 1/(6 kappa) -> 1/(k kappa), z0 = 17/50 -> (17/50)(6/k).
'off' names substitutions kept at their sextic value (ablation). Not derived from a cubic proof.
Usage: python modelk.py K [off1,off2,...]"""
import sys, numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
k = float(sys.argv[1]); off = set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] != '-' else set()
kk = lambda name: 6.0 if name in off else k
z0 = (17/50)*6/kk('z0')
KMIN = 0.75  # count-lemma clamp; kappatest.py lowers it
RR = np.linspace(0, 1, 401); TS = np.linspace(1, 1.5, 14)
def count_t(t, d, x, s0, sup, k0=1.0, slots=True):
    kap = max(KMIN, 2*s0 - 1); c = 1/(kk('cap')*kap); q = x*d; ke = kk('elong')
    r = t - 0.5 + 0.5*RR; m = t - r
    el = np.where(r <= 1/k0, 1.0, (1 + (ke-1)*k0*r)/ke) - d*r
    zi = np.minimum((1-r)/2, sup) if slots else 0.0
    zp = np.minimum(c*(1-2*m), sup) if slots else 0.0
    sh = np.minimum(1 - d*r - 2*q*zi, 1 - 2*d*m - 2*q*zp)
    return float(np.max(np.where(r >= 1, el, sh)))
def Rstar(d, x, s0, sup, slots):
    best = min(count_t(t, d, x, s0, sup, 1.0, slots) for t in TS)
    res = minimize_scalar(lambda t: count_t(t, d, x, s0, sup, 1.0, slots), bounds=(1, 1.5),
                          method='bounded', options={'xatol': 1e-5})
    return min(best, res.fun)
def check(b, lx, ell, s0, nd=14, nx=4):
    ly = lx + b; M = lx + ly; h = 1 - lx + ell; gam = 1/kk('signal')
    C = lx/2 + s0 - 1 + gam*h
    def E(dd, dl, x, R):
        a = (1 + dl)/2; q = x*dl
        return a - s0 + h*(z0 - gam) - a*ly - ell/2 + q*ell + dd*(R + dl/2 - z0)
    loss = lambda dd: max(0.0, (1 + 3*(ell-dd) - 2*(M-2*dd))/4)
    out = {}
    out['low'] = lx/2 + b/(2*kk('gram')) + max(-dd + loss(dd)/2 for dd in np.linspace(0, ell, 40)) - C
    out['len1'] = -(lx - ell); out['len2'] = -(ly - ell - (2 - 1/kk('gram'))*b)
    out['small'] = h*(z0 - gam) - ly/2; out['struct'] = M + ell - 1
    kap = 2*s0 - 1; sup = ell/h; wh = wi = -9
    for dl in np.linspace(1/50, max(kap, 1/50), nd):
        for x in np.linspace(0, 0.5, nx):
            wh = max(wh, E(h, dl, x, Rstar(dl, x, s0, sup, True)))
            wi = max(wi, E(0.5, dl, x, Rstar(dl, x, s0, sup, False)))
    out['high_h'] = wh; out['high_mid'] = wi
    out['floor'] = max(E(h, 1/50, 0.5, 1.0), E(0.5, 1/50, 0.5, 1.0))
    out['mid_t1'] = max(E(0.5, dl, 0.5, 76/75 - 2*dl/3) for dl in np.linspace(1/50, max(kap, 1/50), nd))
    return out
def worst(p, s0): return max(check(*p, s0).values()) if p[0] >= 0 else 1 + abs(p[0])
if __name__ == '__main__':
  lo, hi, best = 0.60, 0.99, None
  for it in range(8):
      mid = (lo + hi)/2
      r = differential_evolution(worst, [(0, 0.4), (0.05, 0.7), (0, 0.5)], args=(mid,), seed=1,
                                 maxiter=20, popsize=8, tol=1e-8)
      if r.fun < -1e-6: hi, best = mid, r.x
      else: lo = mid
  tag = f"k={k:g} off={','.join(sorted(off)) or '-'}"
  if best is None: print("RESULT", tag, "infeasible below 0.99"); sys.exit()
  v = check(*best, hi)
  print("RESULT", tag, "s0<=", round(hi, 4), np.round(best, 3), "binding:", [n for n, a in v.items() if a > -3e-3])
