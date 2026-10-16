"""model78 plus the paper's stated intermediate bound (Sec. Frequency ranges, line ~16148):
for d<=1/2 no slots, t=1, R = 76/75 - 2 delta/3, q = delta/2, worst d = 1/2. Conservative variant."""
import numpy as np
from model78 import check as check0, E
def check(g, k0=1.0, nd=24, nx=6):
    out = check0(g, k0, nd, nx)
    kap = 2*g['s0'] - 1
    out['mid_t1'] = max(E(0.5, dl, 0.5, 76/75 - 2*dl/3, g) for dl in np.linspace(1/50, max(kap, 1/50), nd))
    return out
