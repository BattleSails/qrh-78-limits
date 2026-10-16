"""
Zero spectroscopy for the row u = 1 (item 3 of the caveat plan).

A_1(D) = sum_n mu(n) W(N n / D) over ideals n of Z[w] prime to 6 equals
sum_m a(m) W(m/D), where sum a(m) m^-s = 1/zeta_K^(6)(s),
zeta_K^(6)(s) = zeta(s) L(s, chi_-3) (1 - 4^-s)(1 - 3^-s).
a is multiplicative, supported on m prime to 6:
  p = 1 mod 3: a(p) = -2, a(p^2) = 1;   p = 2 mod 3: a(p) = 0, a(p^2) = -1;  a(p^k) = 0, k >= 3.

Explicit-formula prediction (no fitted parameters):
  A_1(D)/sqrt(D) ~ sum_{gamma > 0} 2 Re( What(rho) D^{i gamma} / zeta_K^(6)'(rho) ),
over zeros rho = 1/2 + i gamma of zeta and of L(s, chi_-3), What(s) = int W(y) y^{s-1} dy.

  python spectro.py
"""
import os
import sys

import mpmath as mp
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, squarefree_ideals
from family import W

MMAX = 20_000_000
GMAX = float(os.environ.get("GMAX", "60"))


def coefficients(M):
    a = np.zeros(M + 1, dtype=np.int64)
    a[1] = 1
    spf = np.zeros(M + 1, dtype=np.int64)
    for i in range(2, M + 1):
        if spf[i] == 0:
            spf[i::i][spf[i::i] == 0] = i
    return _fill(a, spf, M)


@njit(cache=True)
def _fill(a, spf, M):
    for m in range(2, M + 1):
        p = spf[m]
        k, r = 0, m
        while r % p == 0:
            r //= p
            k += 1
        if p == 2 or p == 3 or k >= 3:
            a[m] = 0
            continue
        if p % 3 == 1:
            ap = -2 if k == 1 else 1
        else:
            ap = 0 if k == 1 else -1
        a[m] = ap * a[r]
    return a


@njit(cache=True)
def A1_on_grid(a, Ds, wtab, wn):
    out = np.zeros(len(Ds))
    for i in range(len(Ds)):
        D = Ds[i]
        lo, hi = int(np.ceil(D)), int(np.floor(2 * D))
        s = 0.0
        for m in range(lo, hi + 1):
            if a[m] != 0:
                x = m / D                                   # in [1, 2]
                j = (x - 1.0) * wn
                k = int(j)
                if k >= wn:
                    k = wn - 1
                f = j - k
                s += a[m] * (wtab[k] * (1 - f) + wtab[k + 1] * f)
        out[i] = s
    return out


def check_against_ideals(X=3000):
    P = prime_ideals(X)
    norms, mu, ptr, idx = squarefree_ideals(P, 1, X)
    direct = np.zeros(X + 1, dtype=np.int64)
    np.add.at(direct, norms, mu)
    a = coefficients(X)
    ok = np.array_equal(direct[1:], a[1:X + 1])
    print(f"a(m) = sum over ideals of norm m of mu: {'agrees' if ok else 'DISAGREES'} for m <= {X}")
    return ok


def zeros_L_chi3(gmax):
    """Zeros of L(s, chi_-3) on the critical line up to height gmax (sign changes of Z)."""
    chi = [0, 1, -1]

    def Lambda_phase(t):
        s = mp.mpf(0.5) + 1j * t
        # completed L: (3/pi)^{s/2} Gamma((s+1)/2) L(s); real on the line up to a constant phase
        return (3 / mp.pi) ** (s / 2) * mp.gamma((s + 1) / 2) * mp.dirichlet(s, chi)

    ts = np.arange(0.5, gmax + 0.25, 0.05)
    vals = [Lambda_phase(t) for t in ts]
    ph = mp.arg(vals[0])
    Z = [float(mp.re(v * mp.exp(-1j * ph))) for v in vals]
    zs = []
    for t0, t1, z0, z1 in zip(ts[:-1], ts[1:], Z[:-1], Z[1:]):
        if z0 == 0 or z0 * z1 < 0:
            g = mp.findroot(lambda t: mp.re(Lambda_phase(t) * mp.exp(-1j * ph)), (t0, t1), solver="anderson")
            zs.append(float(g))
    return zs


def prediction(Ds, gmax):
    mp.mp.dps = 30
    chi = [0, 1, -1]
    zk = lambda s: mp.zeta(s) * mp.dirichlet(s, chi) * (1 - 4 ** (-s)) * (1 - 3 ** (-s))
    Wm = lambda y: mp.e * mp.exp(-1 / (1 - (2 * y - 3) ** 2)) if 1 < y < 2 else 0
    zeta_zeros = []
    n = 1
    while True:
        g = float(mp.im(mp.zetazero(n)))
        if g > gmax:
            break
        zeta_zeros.append(g)
        n += 1
    L_zeros = zeros_L_chi3(gmax)
    terms = []
    for g, src in [(g, "zeta") for g in zeta_zeros] + [(g, "L(chi_-3)") for g in L_zeros]:
        rho = mp.mpf(0.5) + 1j * g
        What = mp.quad(lambda y: Wm(y) * y ** (rho - 1), [1, 1.5, 2])
        dz = mp.diff(zk, rho)
        terms.append((g, src, complex(What / dz)))
    terms.sort()
    t = np.log(Ds)
    pred = np.zeros(len(Ds))
    for g, src, c in terms:
        pred += 2 * np.real(c * np.exp(1j * g * t))
    return pred, terms


if __name__ == "__main__":
    if not check_against_ideals():
        sys.exit(1)
    a = coefficients(MMAX)
    Ds = np.logspace(2, np.log10(MMAX / 2), 3000)
    wn = 4000
    wtab = W(np.linspace(1.0, 2.0, wn + 1))
    A = A1_on_grid(a, Ds, wtab, wn)
    meas = A / np.sqrt(Ds)
    pred, terms = prediction(Ds, GMAX)
    print(f"{len(terms)} zeros with 0 < gamma <= {GMAX}; largest |coefficient| by height:")
    for g, src, c in terms[:12]:
        print(f"   gamma = {g:8.4f}  ({src:9s})  |c| = {abs(c):.4f}")
    print(f"   ... smallest |c| among the top-height zeros: {min(abs(c) for g, s, c in terms[-5:]):.2e}")
    for lo in [1e2, 1e4, 1e6]:
        m = Ds >= lo
        rms = np.sqrt(np.mean(meas[m] ** 2))
        err = np.sqrt(np.mean((meas[m] - pred[m]) ** 2))
        corr = np.corrcoef(meas[m], pred[m])[0, 1]
        print(f"D >= {lo:8.0e}: rms(A_1/sqrt D) = {rms:.4f}, rms(measured - predicted) = {err:.4f}, "
              f"relative error {err / rms:.3%}, correlation {corr:.6f}")
    np.savez("runs/spectro_u1.npz", Ds=Ds, meas=meas, pred=pred,
             gam=np.array([g for g, s, c in terms]), src=np.array([s for g, s, c in terms]),
             coef=np.array([c for g, s, c in terms]))
