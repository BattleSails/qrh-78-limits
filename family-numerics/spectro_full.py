"""
Add the Re s = 0 terms to the explicit-formula prediction of spectro.py.

1/zeta_K^(6)(s) has poles on Re s = 0 from the removed Euler factors (1-4^-s)(1-3^-s):
a double pole at s = 0 and simple poles at s = 2 pi i k/ln 4 and s = 2 pi i k/ln 3 (k != 0).
Their contribution to A_1(D) is O(log D), i.e. O(log D / sqrt D) after normalisation.
Residues are computed numerically with mpmath (contour integrals around each pole).

  GMAX=100 python spectro_full.py
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.mp.dps = 30
chi = [0, 1, -1]
Wm = lambda y: mp.e * mp.exp(-1 / (1 - (2 * y - 3) ** 2)) if 1 < y < 2 else 0
What = lambda s: mp.quad(lambda y: Wm(y) * y ** (s - 1), [1, 1.5, 2])
inv_zk = lambda s: 1 / (mp.zeta(s) * mp.dirichlet(s, chi) * (1 - 4 ** (-s)) * (1 - 3 ** (-s)))


def residue(f, s0, r=1e-3, n=64):
    """(1/2 pi i) contour integral of f around s0 (trapezoid rule on a small circle)."""
    tot = 0
    for j in range(n):
        th = 2 * mp.pi * j / n
        z = s0 + r * mp.exp(1j * th)
        tot += f(z) * r * mp.exp(1j * th)
    return tot / n


def boundary_terms(Ds, kmax=6):
    """Return the Re s = 0 contribution to A_1(D) on the grid Ds (unnormalised)."""
    t = np.log(Ds)
    out = np.zeros(len(Ds))
    # double pole at s = 0: residue of What(s) D^s / zeta_K(s) = c1 + c0 log D
    g = lambda s: What(s) * inv_zk(s)
    c_m2 = residue(lambda s: g(s) * s, 0)        # coefficient of s^-2 in the Laurent series
    c_m1 = residue(g, 0)                          # coefficient of s^-1
    out += float(mp.re(c_m1)) + float(mp.re(c_m2)) * t
    # simple poles at 2 pi i k / ln q, q = 4, 3, paired with their conjugates
    for q in (4, 3):
        for k in range(1, kmax + 1):
            s0 = 2j * mp.pi * k / mp.log(q)
            r = residue(lambda s: What(s) * inv_zk(s), s0)
            out += 2 * np.real(complex(r) * np.exp(1j * float(mp.im(s0)) * t))
    return out


if __name__ == "__main__":
    d = np.load("runs/spectro_u1.npz")
    Ds, meas, pred = d["Ds"], d["meas"], d["pred"]
    bt = boundary_terms(Ds) / np.sqrt(Ds)
    for lo in [1e2, 1e3, 1e4, 1e6]:
        m = Ds >= lo
        rms = np.sqrt(np.mean(meas[m] ** 2))
        e0 = np.sqrt(np.mean((meas[m] - pred[m]) ** 2))
        e1 = np.sqrt(np.mean((meas[m] - pred[m] - bt[m]) ** 2))
        c1 = np.corrcoef(meas[m], pred[m] + bt[m])[0, 1]
        print(f"D >= {lo:7.0e}: residual zeros only {e0 / rms:7.3%} -> with Re s = 0 terms {e1 / rms:7.3%}"
              f"   (correlation {c1:.6f})")
    np.savez("runs/spectro_u1_full.npz", Ds=Ds, meas=meas, pred=pred, bt=bt)
