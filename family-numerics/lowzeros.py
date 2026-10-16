"""
Small gaps and small |L'(rho)| over the family (zeros from famspec.py, 1 < N(u) <= 200).

For each row: zeros rho = 1/2 + i gamma (|gamma| <= 50) and residues c = What(rho)/(L'(rho) E(rho));
|L'(rho) E(rho)| = |What(rho)| / |c|. Unfolding by the density (1/2 pi) log(q (gamma/2 pi)^2),
q = 3 N(conductor) the conductor over Q. GUE predicts Pr(s < eps) ~ eps^3 for the normalised
nearest-neighbour gap and Pr(|L'|/log < eps) ~ eps^3; Poisson (factored rows) ~ eps.

  python lowzeros.py
"""
import re

import numpy as np
from scipy.integrate import quad

j = np.load("runs/famspec200_joint.npz")
inv = np.load("runs/famspec200_factored.npy")
y = np.linspace(1, 2, 4001)[1:-1]
t = 2 * y - 3
Wy = np.exp(1 - 1 / (1 - t ** 2))


def What(g):
    s = 0.5 + 1j * g[:, None]
    return np.trapezoid(Wy * y ** (s - 1), y, axis=1)


S = {False: [], True: []}
Lp = {False: [], True: []}
for x, yy, fac in zip(j["ux"], j["uy"], inv):
    fn = f"runs/fam/rowspec_{x}_{yy}.txt"
    q = 3 * int(re.search(r"conductor norm (\d+)", open(fn + ".info").read()).group(1))
    d = np.array([[float(v) for v in l.replace(" e", "e").split()] for l in open(fn) if l.strip()])
    on = np.abs(d[:, 0] - 0.5) < 1e-9
    g, c = d[on, 1], np.abs(d[on, 2] + 1j * d[on, 3])
    o = np.argsort(g)
    g, c = g[o], c[o]
    dens = lambda h: np.log(q * (np.maximum(np.abs(h), 1.0) / (2 * np.pi)) ** 2) / (2 * np.pi)
    gaps = np.diff(g) * dens((g[1:] + g[:-1]) / 2)
    nn = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])     # nearest-neighbour gap
    keep = (np.abs(g) > 3) & (np.abs(g) < 45)
    S[bool(fac)].append(nn[keep])
    lp = np.abs(What(g)) / c / (2 * np.pi * dens(g))                # |L'| in units of the mean density
    Lp[bool(fac)].append(lp[keep])
for fac in (False, True):
    s = np.concatenate(S[fac])
    l = np.concatenate(Lp[fac])
    lab = "factored rows" if fac else "generic rows"
    print(f"{lab}: {len(s)} zeros; mean nn gap {s.mean():.3f}")
    for name, v in (("nn gap s", s / s.mean()), ("|L'|", l / np.median(l))):
        eps = np.array([0.05, 0.1, 0.2, 0.4])
        F = np.array([np.mean(v < e) for e in eps])
        sl = np.polyfit(np.log(eps[F > 0]), np.log(F[F > 0]), 1)[0] if (F > 0).sum() > 1 else np.nan
        print(f"   {name:9s}: Pr(< eps) at eps = 0.05/0.1/0.2/0.4: " + " ".join(f"{f:.2e}" for f in F)
              + f"   log-log slope {sl:.2f}")
np.savez("runs/lowzeros.npz", s_gen=np.concatenate(S[False]), s_fac=np.concatenate(S[True]),
         l_gen=np.concatenate(Lp[False]), l_fac=np.concatenate(Lp[True]))
