import numpy as np
c = 2/9
NT, NR = 500, 200
def R_lb(dl, x, sup, slots, k0=1.0):
    q = x*dl
    t = 1 + np.arange(NT + 1)/(2*NT)
    t1, t2 = t[:-1, None], t[1:, None]
    rho = np.arange(NR + 1)[None, :]/NR
    m = (1 - rho)/2; r1, r2 = t1 - 0.5 + rho/2, t2 - 0.5 + rho/2
    ra = np.maximum(r1, 1.0); el = np.where(ra <= 1/k0, 1.0, (1 + 5*k0*ra)/6) - dl*r2
    rb = np.minimum(r2, 1.0)
    zi = np.minimum((1 - r1)/2, sup) if slots else 0.0
    zp = np.minimum(c*(1 - 2*m), sup) if slots else 0.0
    sh = np.minimum(1 - dl*rb - 2*q*zi, 1 - 2*dl*m - 2*q*zp)
    lb = np.where(r1 >= 1, el, np.where(r2 < 1, sh, np.minimum(el, sh)))
    return float(np.min(np.max(lb, axis=1)))
