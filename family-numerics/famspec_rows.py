"""
Per-row check for famspec.py: average |A_u(D)|^2 / D over 300 log-spaced D in [1e3, 1e6]
(sieve as in rowspec.py) against the spectral mass M_u from PARI.

  python famspec_rows.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals
from rowspec import psi_at_primes, _fill, A_on_grid

M = 2_000_000
d = np.load("runs/famspec.npz")
ux, uy, un, Mspec = d["ux"], d["uy"], d["un"], d["Mspec"]
P = prime_ideals(M)
spf = np.zeros(M + 1, dtype=np.int64)
for i in range(2, int(M ** 0.5) + 1):
    if spf[i] == 0:
        spf[i * i::i][spf[i * i::i] == 0] = i
spf[spf == 0] = np.arange(M + 1)[spf == 0]
split = P["kind"] == 0
Ds = np.logspace(3, np.log10(M / 2), 300)
avg = np.zeros(len(ux))
for k, (x, y) in enumerate(zip(ux, uy)):
    ps = psi_at_primes(P, int(x), int(y))
    f1 = np.zeros(M + 1, dtype=np.complex128)
    f2 = np.zeros(M + 1, dtype=np.complex128)
    f2[P["p"][~split]] = -ps[~split]
    sp, sv = P["p"][split], ps[split]              # split primes come in adjacent pairs
    np.add.at(f1, sp, -sv)
    prod = np.ones(M + 1, dtype=np.complex128)
    np.multiply.at(prod, sp, sv)
    f2[sp] = prod[sp]
    a = np.zeros(M + 1, dtype=np.complex128)
    a[1] = 1
    a = _fill(a, spf, M, f1, f2)
    A = A_on_grid(a, Ds)
    avg[k] = np.mean(np.abs(A) ** 2 / Ds)
np.savez("runs/famspec_rows.npz", ux=ux, uy=uy, un=un, Mspec=Mspec, avg=avg, Ds=Ds)
r = avg / Mspec
print(f"{len(ux)} rows: mean M_u = {Mspec.mean():.4f}, mean of log-D average |A_u|^2/D = {avg.mean():.4f}")
print(f"per-row correlation {np.corrcoef(Mspec, avg)[0, 1]:.3f}; ratio avg/M_u quantiles (10/50/90%): "
      + " ".join(f"{v:.3f}" for v in np.quantile(r, [0.1, 0.5, 0.9])))
for j in np.argsort(-np.abs(np.log(r)))[:8]:
    print(f"  u = ({ux[j]:3d},{uy[j]:3d}) N = {un[j]:3d}: M_u = {Mspec[j]:.3f}, average = {avg[j]:.3f}")
