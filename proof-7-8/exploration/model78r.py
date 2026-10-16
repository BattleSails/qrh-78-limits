"""model78 with a cancelling floor: rows whose zeros all have delta <= d1 are treated on a common
contour at a=(1+d1)/2 with count exponent rho (hypothetical cross-row cancellation); detector bins
only for delta in [d1, kappa]. cons=True adds the paper's t=1 intermediate bound for delta>=d1."""
import numpy as np
from model78 import Rstar, E, z0
def check(g, k0=1.0, rho=1.0, cons=False, nd=24, nx=6):
    b, lx, ell, s0, d1 = g['b'], g['lx'], g['ell'], g['s0'], g['d1']
    ly = lx + b; M = lx + ly; h = 1 - lx + ell
    out = {}
    loss = lambda dd: max(0.0, (1 + 3*(ell-dd) - 2*(M-2*dd))/4)
    low = lx/2 + b/12 + max(-dd + loss(dd)/2 for dd in np.linspace(0, ell, 50))
    out['low'] = low - (lx/2 + s0 - 1 + h/6)
    out['len1'] = -(lx - ell); out['len2'] = -(ly - ell - 11*b/6)
    out['small'] = h*(z0 - 1/6) - ly/2
    out['struct'] = M + ell - 1
    out['d1lo'] = 1/50 - d1
    kap = 2*s0 - 1; sup = ell/h
    wh = wi = wt = -9
    for dl in np.linspace(d1, max(kap, d1), nd):
        for x in np.linspace(0, 0.5, nx):
            wh = max(wh, E(h, dl, x, Rstar(dl, x, s0, sup, k0, True), g))
            wi = max(wi, E(0.5, dl, x, Rstar(dl, x, s0, sup, k0, False), g))
        if cons: wt = max(wt, E(0.5, dl, 0.5, 76/75 - 2*dl/3, g))
    out['high_h'] = wh; out['high_mid'] = wi
    if cons: out['mid_t1'] = wt
    out['floor'] = max(E(h, d1, 0.5, rho, g), E(0.5, d1, 0.5, rho, g))
    return out
