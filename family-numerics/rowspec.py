"""
Zero spectroscopy for rows u != 1 (needs PARI/GP; rowchar.gp does the L-function side).

A_u(D) = sum_n mu(n) (u/n)_6 W(N n / D) = sum_m a_u(m) W(m / D), where a_u is the
multiplicative function on Z with  sum a_u(m) m^-s = prod_{P prime to 6u} (1 - psi(P) N P^-s):
  p = 1 mod 3, P, P' above p:  a(p) = -(psi(P) + psi(P')),  a(p^2) = psi(P) psi(P');
  q = 2 mod 3, q >= 5:         a(q^2) = -psi(q);           psi(P) = 0 if P | u.
psi(P) = (u/P)_6 is computed here with the same sextic_exponent as the family code, so the
measured side is independent of PARI. PARI identifies the primitive Hecke character from its
values at primes up to 3000 and returns its zeros rho and the residues
What(rho) / L^S'(rho), plus the Re s = 0 poles. Prediction (no fitted parameters):
  A_u(D) ~ sum over poles s0 of  c(s0) D^s0.

  python rowspec.py MMAX GMAX x1,y1 x2,y2 ...
"""
import os
import subprocess
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals
from family import W, family_sums
from spectro import A1_on_grid

GP = os.environ.get("GP", "gp")  # set the GP environment variable if gp is not on PATH
Z6 = np.exp(1j * np.pi * np.arange(6) / 3)


def psi_at_primes(P, x, y):
    e = np.array([sextic_exponent(x, y, P["kind"][i], P["p"][i], P["r"][i], P["pw"][i])
                  for i in range(len(P["p"]))])
    return np.where(e >= 0, Z6[np.maximum(e, 0)], 0)


@njit(cache=True)
def _fill(a, spf, M, f1, f2):
    # f1[p] = a(p), f2[p] = a(p^2); a(p^k) = 0 for k >= 3 (all Euler factors have degree <= 2)
    for m in range(2, M + 1):
        p = spf[m]
        k, r = 0, m
        while r % p == 0:
            r //= p
            k += 1
        if k == 1:
            a[m] = f1[p] * a[r]
        elif k == 2:
            a[m] = f2[p] * a[r]
        else:
            a[m] = 0
    return a


def coefficients(x, y, M):
    P = prime_ideals(M)
    ps = psi_at_primes(P, x, y)
    f1 = np.zeros(M + 1, dtype=np.complex128)
    f2 = np.zeros(M + 1, dtype=np.complex128)
    split = {}
    for i in range(len(ps)):
        p = int(P["p"][i])
        if P["kind"][i] == 0:
            split.setdefault(p, []).append(ps[i])
        else:
            f2[p] = -ps[i]
    for p, (s1, s2) in split.items():
        f1[p] = -(s1 + s2)
        f2[p] = s1 * s2
    spf = np.zeros(M + 1, dtype=np.int64)
    for i in range(2, int(M ** 0.5) + 1):
        if spf[i] == 0:
            spf[i * i::i][spf[i * i::i] == 0] = i
    spf[spf == 0] = np.arange(M + 1)[spf == 0]
    a = np.zeros(M + 1, dtype=np.complex128)
    a[1] = 1
    return _fill(a, spf, M, f1, f2)


def check_against_family(x, y, a, D=3000):
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    coef = mu * W(norms / D)
    Are, Aim, _ = family_sums(np.array([x]), np.array([y]), P["kind"], P["p"], P["r"], P["pw"],
                              coef, ptr, idx)
    m = np.arange(D, 2 * D + 1)
    direct = np.sum(a[m] * W(m / D))
    err = abs(direct - (Are[0] + 1j * Aim[0]))
    print(f"  A_u({D}) family code {Are[0]:+.6f}{Aim[0]:+.6f}i, sieve {direct:+.6f}: |diff| = {err:.1e}")
    return err < 1e-8


def A_on_grid(a, Ds):
    wn = 4000
    wtab = W(np.linspace(1.0, 2.0, wn + 1))
    return A1_on_grid(a.real.copy(), Ds, wtab, wn) + 1j * A1_on_grid(a.imag.copy(), Ds, wtab, wn)


def pari_poles(x, y, gmax):
    fn = f"runs/rowspec_{x}_{y}.txt"
    cmd = f'read("rowchar.gp"); rowspec({x}, {y}, {gmax})'
    if os.path.exists(fn):
        os.remove(fn)
    r = subprocess.run([GP, "-q", "-s", "2G"], input=cmd, capture_output=True, text=True)
    print("  " + "\n  ".join(l for l in r.stdout.splitlines() if "Warning" not in l))
    with open(fn) as f:                             # PARI writes '1.0 e-5'
        d = np.array([[float(v) for v in l.replace(" e", "e").split()] for l in f if l.strip()])
    return d[:, 0] + 1j * d[:, 1], d[:, 2] + 1j * d[:, 3]


def main(M, gmax, rows):
    Ds = np.logspace(2, np.log10(M / 2), 2000)
    out = {}
    for x, y in rows:
        t0 = time.time()
        print(f"u = {x} + {y} w (norm {x * x - x * y + y * y}):")
        a = coefficients(x, y, M)
        if not check_against_family(x, y, a):
            sys.exit(1)
        meas = A_on_grid(a, Ds) / np.sqrt(Ds)
        s0, c = pari_poles(x, y, gmax)
        on_line = np.abs(s0.real - 0.5) < 1e-9
        t = np.log(Ds)
        pz = (c[on_line, None] * np.exp(1j * s0.imag[on_line, None] * t)).sum(0)
        logt = s0.real == -1                            # coefficient of log D (double pole at 0)
        rest = ~on_line & ~logt
        pb = ((c[rest, None] * np.exp(s0[rest, None] * t)).sum(0) + c[logt].sum() * t) / np.sqrt(Ds)
        for lo in [1e2, 1e4, 1e6]:
            m = Ds >= lo
            rms = np.sqrt(np.mean(np.abs(meas[m]) ** 2))
            e0 = np.sqrt(np.mean(np.abs(meas[m] - pz[m]) ** 2))
            e1 = np.sqrt(np.mean(np.abs(meas[m] - pz[m] - pb[m]) ** 2))
            cor = np.abs(np.vdot(meas[m], pz[m] + pb[m])) / np.linalg.norm(meas[m]) / np.linalg.norm(pz[m] + pb[m])
            print(f"  D >= {lo:7.0e}: rms |A_u|/sqrt D = {rms:.4f}; residual zeros only {e0 / rms:7.3%}, "
                  f"+ Re s = 0 terms {e1 / rms:7.3%}; |correlation| {cor:.6f}")
        k = np.argsort(-np.abs(c[on_line]))[:5]
        print("  largest zero terms: " + ", ".join(f"g={s0[on_line][j].imag:+.3f} |c|={abs(c[on_line][j]):.3f}"
                                                  for j in k))
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
        out[f"{x}_{y}"] = (meas, pz, pb)
        np.savez(f"runs/rowspec_{x}_{y}.npz", Ds=Ds, meas=meas, pz=pz, pb=pb, s0=s0, c=c)


if __name__ == "__main__":
    M, gmax = int(float(sys.argv[1])), float(sys.argv[2])
    rows = [tuple(int(v) for v in s.split(",")) for s in sys.argv[3:]]
    main(M, gmax, rows)
