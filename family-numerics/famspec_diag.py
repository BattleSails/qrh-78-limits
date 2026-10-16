"""
Per-row quantities for every u with 1 < N(u) <= HMAX (rows grouped by norm):
  avg   = mean of |A_u(D)|^2 / D over 300 log-spaced D in [1e3, 1e6]
  ndiag = sum_m |a_u(m)|^2 W(m/D)^2 / D   (diagonal of the norm-grouped sum; mean over 5 D)
  mass  = sum_{(n,u)=1} mu(n)^2 W(N n/D)^2 / D  (per-ideal random model; mean over the same D)
a_u is as in rowspec.py. Compared with the spectral mass M_u from famspec.py.

  python famspec_diag.py HMAX
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, elements_upto
from family import W
from rowspec import psi_at_primes, _fill, A_on_grid

M = 2_000_000
hmax = int(sys.argv[1])
ux, uy, un = elements_upto(hmax)
k = un > 1
ux, uy, un = ux[k], uy[k], un[k]
P = prime_ideals(M)
spf = np.zeros(M + 1, dtype=np.int64)
for i in range(2, int(M ** 0.5) + 1):
    if spf[i] == 0:
        spf[i * i::i][spf[i * i::i] == 0] = i
spf[spf == 0] = np.arange(M + 1)[spf == 0]
split = P["kind"] == 0
Ds = np.logspace(3, np.log10(M / 2), 300)
Dd = np.logspace(5, np.log10(M / 2), 5)


def coeffs(ps):
    f1 = np.zeros(M + 1, dtype=np.complex128)
    f2 = np.zeros(M + 1, dtype=np.complex128)
    f2[P["p"][~split]] = -ps[~split]
    sp, sv = P["p"][split], ps[split]
    np.add.at(f1, sp, -sv)
    prod = np.ones(M + 1, dtype=np.complex128)
    np.multiply.at(prod, sp, sv)
    f2[sp] = prod[sp]
    a = np.zeros(M + 1, dtype=np.complex128)
    a[1] = 1
    return _fill(a, spf, M, f1, f2)


avg, ndiag, mass = (np.zeros(len(ux)) for _ in range(3))
for j, (x, y) in enumerate(zip(ux, uy)):
    ps = psi_at_primes(P, int(x), int(y))
    a = coeffs(ps)
    avg[j] = np.mean(np.abs(A_on_grid(a, Ds)) ** 2 / Ds)
    # per-ideal mass: same sieve with |psi| in place of psi and mu^2 (no sign) -> coefficients of
    # prod (1 + |psi(P)| N P^-s); obtained from coeffs() with -|psi| and taking absolute values
    b = np.abs(coeffs(-np.abs(ps)))
    a2 = np.abs(a) ** 2
    for D in Dd:
        m = np.arange(int(np.ceil(D)), int(2 * D) + 1)
        w2 = W(m / D) ** 2
        ndiag[j] += np.sum(a2[m] * w2) / D / len(Dd)
        mass[j] += np.sum(b[m] * w2) / D / len(Dd)
    if j % 100 == 0:
        print(j, flush=True)
np.savez(f"runs/famspec_diag{hmax}.npz", ux=ux, uy=uy, un=un, avg=avg, ndiag=ndiag, mass=mass)
