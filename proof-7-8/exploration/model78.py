"""Reduced constraint model of Part II of the OAI 7/8 proof (paper.tex, Sept 30 2026).

Geometry: b, l_x, ell free;  l_y = l_x + b,  M = l_x + l_y,  h = 1 - l_x + ell.
Constraints modelled (paper locations in comments):
  low   : l_x/2 + b/12 + max_d[-d + loss(d)/2] <= C(s0) = l_x/2 + s0 - 1 + h/6     (8564-8642, 8308)
          loss(d) = ((1 + 3(ell-d) - 2(M-2d))/4)_+                                   (8308-8325)
          l_x - ell > 0,  l_y - ell - 11b/6 >= 0                                      (8594-8597)
  high  : E(d) = a - s0 + h(z0-1/6) - a l_y - ell/2 + q ell + d(R + delta/2 - z0) < 0 (eq stage-high-exponent)
          adaptive range d in [1/2, h] (worst d = h), intermediate d <= 1/2 (no slots),
          floor bin delta0 = 1/50 with R = 1, small rows h(z0-1/6) - l_y/2 < 0.
  counts: Prop 19.2 with plain capacity c(1-2m), c = 1/(6 kappa), kappa = 2 s0 - 1,
          inverse capacity (1-r)/2, both capped by supply ell/h; long range e(r) from
          Lemma 17.6, optionally improved by MS(kappa) for kappa in [k0, 1].
Not modelled: anything else in the 16k-line proof. A feasible point here is a candidate only.
"""
import numpy as np
z0 = 17/50
def e_long(r, k0):
    if r <= 1/k0: return 1.0
    return (1 + 5*k0*r)/6
RR = np.linspace(0, 1, 801)
def count_t(t, d, x, s0, sup, k0, slots=True):
    kap = max(0.75, 2*s0 - 1); c = 1/(6*kap); q = x*d  # Lemma 18.x needs kappa>=3/4
    r = t - 0.5 + 0.5*RR
    m = t - r
    el = np.where(r <= 1/k0, 1.0, (1 + 5*k0*r)/6) - d*r
    zi = np.minimum((1-r)/2, sup) if slots else 0.0
    zp = np.minimum(c*(1-2*m), sup) if slots else 0.0
    sh = np.minimum(1 - d*r - 2*q*zi, 1 - 2*d*m - 2*q*zp)
    return float(np.max(np.where(r >= 1, el, sh)))
TS = np.linspace(1, 1.5, 26)
from scipy.optimize import minimize_scalar
def Rstar(d, x, s0, sup, k0, slots=True):
    best = min(count_t(t, d, x, s0, sup, k0, slots) for t in TS)
    res = minimize_scalar(lambda t: count_t(t, d, x, s0, sup, k0, slots),
                          bounds=(1, 1.5), method='bounded', options={'xatol': 1e-6})
    return min(best, res.fun)
def E(dd, a_delta, x, R, g):
    b, lx, ell, s0 = g['b'], g['lx'], g['ell'], g['s0']
    ly = lx + b; h = 1 - lx + ell
    a = (1 + a_delta)/2; q = x*a_delta
    return a - s0 + h*(z0 - 1/6) - a*ly - ell/2 + q*ell + dd*(R + a_delta/2 - z0)
def check(g, k0=1.0, nd=24, nx=6, verbose=False):
    b, lx, ell, s0 = g['b'], g['lx'], g['ell'], g['s0']
    ly = lx + b; M = lx + ly; h = 1 - lx + ell
    out = {}
    loss = lambda dd: max(0.0, (1 + 3*(ell-dd) - 2*(M-2*dd))/4)
    low = lx/2 + b/12 + max(-dd + loss(dd)/2 for dd in np.linspace(0, ell, 50))
    out['low'] = low - (lx/2 + s0 - 1 + h/6)
    out['len1'] = -(lx - ell); out['len2'] = -(ly - ell - 11*b/6)
    out['small'] = h*(z0 - 1/6) - ly/2
    out['struct'] = M + ell - 1          # paper uses M + ell = 1 (old-eq:5.1, 8306)
    kap = 2*s0 - 1; sup = ell/h
    worst_h = worst_i = -9
    for dl in np.linspace(1/50, kap, nd):
        for x in np.linspace(0, 0.5, nx):
            Rh = Rstar(dl, x, s0, sup, k0, True)
            worst_h = max(worst_h, E(h, dl, x, Rh, g))
            Ri = Rstar(dl, x, s0, sup, k0, False)
            worst_i = max(worst_i, E(0.5, dl, x, Ri, g))
    fl = 1/50
    worst_f = max(E(h, fl, 0.5, 1.0, g), E(0.5, fl, 0.5, 1.0, g))
    out['high_h'] = worst_h; out['high_mid'] = worst_i; out['floor'] = worst_f
    return out
if __name__ == "__main__":
    g = dict(b=1/8, lx=17/48, ell=1/6, s0=7/8)
    print("paper geometry, k0=1:", {k: round(v, 6) for k, v in check(g).items()})
