"""Exponent-level ledger of the two-branch induction in Lemma plain, order k (light test).
Uncentred threshold theta = 1 - 1/k. For a product of length A in (theta M, M] with slot length z:
  comparison branch: A_comp <= M - A + 2L + 2z < theta M   and   A_comp + (k kappa - 1) z <= M
  centred branch:    A - theta M - s v - (L - v)_+ <= 0 for all v >= 0  (A <= M, positive part 0)
                     i.e. L >= (A - theta M)/min(s, 1)
  slot region:       A + (k kappa - 1) z <= M
Fixed L: one L for every A (the paper's L = M/4). Adaptive L: L may depend on A and z.
Units M = 1. Exact rationals on a grid; returns the worst margin (positive = closes)."""
from fractions import Fraction as F
def margins(k, s, kappa, nA=200, nz=40):
    th = 1 - F(1, k); sm = min(s, 1)
    worst_adapt = None; Lfix_lo = F(0); Lfix_hi = F(10)
    zmax = F(1, k) / (k*kappa - 1) if k*kappa > 1 else F(0)
    for iz in range(nz + 1):
        z = zmax*iz/nz
        Amax = 1 - (k*kappa - 1)*z
        if Amax <= th: continue
        for iA in range(1, nA + 1):
            A = th + (Amax - th)*iA/nA
            lo = (A - th)/sm                                   # centred
            hi = min((th - 1 + A - 2*z)/2, (A - (k*kappa + 1)*z)/2)   # comparison (strict upper)
            m = hi - lo
            worst_adapt = m if worst_adapt is None or m < worst_adapt else worst_adapt
            Lfix_lo = max(Lfix_lo, lo); Lfix_hi = min(Lfix_hi, hi)
    return worst_adapt, Lfix_hi - Lfix_lo
if __name__ == "__main__":
    print("k  s     kappa  adaptive-L margin   fixed-L margin   (margin > 0: the ledger closes)")
    for k in (3, 4, 5, 6):
        for s in (F(2, 3), F(4, k), F(1)):
            for kappa in (F(3, 4), F(1)):
                a, f = margins(k, s, kappa)
                print(f"{k}  {str(s):5} {str(kappa):5}  {float(a):+.4f}            {float(f):+.4f}")
