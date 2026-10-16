"""model78 + 'top dyads on the original side' (heuristic).
Rows u are Poisson frequencies H=u a^6 of the m-sum.  The complete sum over a row dyad U=Z^d is,
by Poisson duality, an original-side sum; the sixth-power-free filter costs a g-sum of length
Z^{(h-d)/6} (trivial bound), so its cost is low_exp + c(h-d)/6 (c=1: trivial heuristic; larger c
allows extra large-sieve loss).  Dyads with that cost <= C(s0) need no zero detection; the
Poisson-side constraints are then evaluated at d* = h - 6 m_low / c instead of d = h, where
m_low = C(s0) - low_exp."""
import numpy as np
from model78 import Rstar, E, z0
def check(g, k0=1.0, c=1.0, cons=False, nd=24, nx=6):
    b, lx, ell, s0 = g['b'], g['lx'], g['ell'], g['s0']
    ly = lx + b; M = lx + ly; h = 1 - lx + ell
    out = {}
    loss = lambda dd: max(0.0, (1 + 3*(ell-dd) - 2*(M-2*dd))/4)
    low = lx/2 + b/12 + max(-dd + loss(dd)/2 for dd in np.linspace(0, ell, 50))
    C = lx/2 + s0 - 1 + h/6
    out['low'] = low - C
    mlow = max(0.0, C - low)
    ds = max(0.5, h - 6*mlow/c)
    out['len1'] = -(lx - ell); out['len2'] = -(ly - ell - 11*b/6)
    out['small'] = h*(z0 - 1/6) - ly/2
    out['struct'] = M + ell - 1
    kap = 2*s0 - 1; sup = ell/ds
    wh = wi = wt = -9
    for dl in np.linspace(1/50, kap, nd):
        for x in np.linspace(0, 0.5, nx):
            wh = max(wh, E(ds, dl, x, Rstar(dl, x, s0, sup, k0, True), g))
            wi = max(wi, E(0.5, dl, x, Rstar(dl, x, s0, sup, k0, False), g))
        if cons: wt = max(wt, E(0.5, dl, 0.5, 76/75 - 2*dl/3, g))
    out['high_top'] = wh; out['high_mid'] = wi
    if cons: out['mid_t1'] = wt
    out['floor'] = max(E(ds, 1/50, 0.5, 1.0, g), E(0.5, 1/50, 0.5, 1.0, g))
    out['dstar'] = ds - h - 1   # informational, always negative
    return out
