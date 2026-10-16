"""model78 with an effective row ceiling h' <= h_max = 1 - l_x + ell (heuristic).
Rows u are the sixth-power-free parts of the Poisson frequency H = u a^6 of the m-sum.  Split the
dual sum by the size of H: the bands q_H > Z^{h'} form, by Poisson duality, a band-filtered copy of
the original-side probe (no sixth-power-free filter is needed), assumed bounded by the low exponent.
The remaining part is the paper's high side with h replaced by h': rows only up to Z^{h'}, and the
principal signal from a-values with q_a^6 <= Z^{h'}, so C(s) uses h'/6.  The low exponent itself
does not depend on h."""
import numpy as np
from model78 import Rstar, E, z0
def check(g, k0=1.0, cons=False, nd=24, nx=6):
    b, lx, ell, s0, f = g['b'], g['lx'], g['ell'], g['s0'], g['f']
    ly = lx + b; M = lx + ly; hmax = 1 - lx + ell; h = f*hmax
    gg = dict(g); gg['lx'] = 1 + ell - h      # E() derives h from lx, ell; keep l_y explicit below
    out = {}
    loss = lambda dd: max(0.0, (1 + 3*(ell-dd) - 2*(M-2*dd))/4)
    low = lx/2 + b/12 + max(-dd + loss(dd)/2 for dd in np.linspace(0, ell, 50))
    out['low'] = low - (lx/2 + s0 - 1 + h/6)
    out['len1'] = -(lx - ell); out['len2'] = -(ly - ell - 11*b/6)
    out['small'] = h*(z0 - 1/6) - ly/2
    out['struct'] = M + ell - 1
    out['f'] = f - 1
    kap = 2*s0 - 1; sup = ell/h
    def EE(d, dl, x, R):
        a = (1 + dl)/2; q = x*dl
        return a - s0 + h*(z0 - 1/6) - a*ly - ell/2 + q*ell + d*(R + dl/2 - z0)
    wh = wi = wt = -9
    for dl in np.linspace(1/50, kap, nd):
        for x in np.linspace(0, 0.5, nx):
            wh = max(wh, EE(h, dl, x, Rstar(dl, x, s0, sup, k0, True)))
            wi = max(wi, EE(0.5, dl, x, Rstar(dl, x, s0, sup, k0, False)))
        if cons: wt = max(wt, EE(0.5, dl, 0.5, 76/75 - 2*dl/3))
    out['high_h'] = wh; out['high_mid'] = wi
    if cons: out['mid_t1'] = wt
    out['floor'] = max(EE(h, 1/50, 0.5, 1.0), EE(0.5, 1/50, 0.5, 1.0))
    return out
