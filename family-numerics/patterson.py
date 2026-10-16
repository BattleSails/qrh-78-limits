"""
Sums of normalised cubic Gauss sums gamma_2 over Z[w] (item 2 of the caveat plan).

  S_p(X)  = sum_{N p <= X, p prime}           gamma_2(p)      (Patterson: ~ c X^{5/6}/log X)
  S_1(X)  = sum_{N n <= X, n squarefree}       gamma_2(n)
  S_mu(X) = sum_{N n <= X, n squarefree} mu(n) conj(gamma_2(n))  (= dual coefficients of the
                                                                 cubic family, = conj(alpha) gamma_2^2)

gamma_2 at split primes: full sum over F_p, enumerated as powers of a primitive root.
gamma_2 at inert primes: full sum over a residue system (gauss.gauss_sums).
gamma_2 at composite n: CRT formula gamma_2(ab) = gamma_2(a) gamma_2(b) chi_a(b)^2 chi_b(a)^2,
which is first checked against brute force (gauss.gauss_sums) for N(n) <= 20000.

  python patterson.py X
"""
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals
import gauss


def primitive_root(p):
    m, fs, d = p - 1, [], 2
    while d * d <= m:
        if m % d == 0:
            fs.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        fs.append(m)
    for g in range(2, p):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fs):
            return g
    raise ValueError(p)


@njit(parallel=True, cache=True)
def split_gamma2(ps, bs, gs, e_g):
    """gamma_2(pi) for split primary pi = (a, b) of prime norm p, via powers of the root g."""
    n = len(ps)
    out_re = np.zeros(n)
    out_im = np.zeros(n)
    re6 = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    im6 = np.array([0.0, 0.8660254037844386, 0.8660254037844386, 0.0,
                    -0.8660254037844386, -0.8660254037844386])
    for t in prange(n):
        p, b, g, eg = ps[t], bs[t], gs[t], e_g[t]
        x = 1
        sr = 0.0
        si = 0.0
        for k in range(p - 1):
            e = (2 * k * eg) % 6                       # chi(x)^2 with x = g^k
            v = (-b * x) % p                           # w-coefficient of x * conj(pi)
            ang = 2.0 * np.pi * v / p
            c, s = np.cos(ang), np.sin(ang)
            sr += re6[e] * c - im6[e] * s
            si += re6[e] * s + im6[e] * c
            x = x * g % p
        out_re[t] = sr / np.sqrt(p)
        out_im[t] = si / np.sqrt(p)
    return out_re + 1j * out_im


def prime_gamma2(P, gens):
    nP = len(P["norm"])
    gam = np.zeros(nP, dtype=np.complex128)
    split = np.nonzero(P["kind"] == 0)[0]
    ps = P["p"][split]
    bs = np.array([gens[i][1] for i in split], dtype=np.int64)
    gs = np.array([primitive_root(int(p)) for p in ps], dtype=np.int64)
    e_g = np.array([sextic_exponent(int(g), 0, 0, int(P["p"][i]), int(P["r"][i]), P["pw"][i])
                    for g, i in zip(gs, split)], dtype=np.int64)
    gam[split] = split_gamma2(ps, bs, gs, e_g)
    for i in np.nonzero(P["kind"] == 1)[0]:
        a, b = gens[i]
        S = gauss.gauss_sums(a, b, int(P["norm"][i]), np.array([i]), P["kind"], P["p"], P["r"],
                             P["pw"], np.array([2], dtype=np.int64))
        gam[i] = (S[0, 0] + 1j * S[0, 1]) / np.sqrt(int(P["norm"][i]))
    return gam


def composite_gamma2(P, gens, gam_p, ptr, idx):
    """gamma_2(n) for squarefree n via the CRT formula, pairwise cubic symbols."""
    n = len(ptr) - 1
    out = np.ones(n, dtype=np.complex128)
    for t in range(n):
        fac = idx[ptr[t]:ptr[t + 1]]
        val = 1.0 + 0j
        e = 0
        for i in fac:
            val *= gam_p[i]
        for x in range(len(fac)):
            for y in range(x + 1, len(fac)):
                i, j = fac[x], fac[y]
                # chi_a(b)^2 chi_b(a)^2 with a = gen(i), b = gen(j)
                e += 2 * sextic_exponent(gens[j][0], gens[j][1], P["kind"][i], P["p"][i], P["r"][i], P["pw"][i])
                e += 2 * sextic_exponent(gens[i][0], gens[i][1], P["kind"][j], P["p"][j], P["r"][j], P["pw"][j])
        out[t] = val * np.exp(1j * np.pi * (e % 6) / 3)
    return out


def check_crt(Xc=20000):
    P = prime_ideals(Xc)
    gens = gauss.primary_generators(P)
    gam_p = prime_gamma2(P, gens)
    norms, mu, ptr, idx = squarefree_ideals(P, 1, Xc)
    g_crt = composite_gamma2(P, gens, gam_p, ptr, idx)
    worst = 0.0
    for t in range(1, len(norms)):
        fac = idx[ptr[t]:ptr[t + 1]]
        a, b = 1, 0
        for i in fac:
            a, b = gauss.mul((a, b), gens[i])
        S = gauss.gauss_sums(a, b, int(norms[t]), fac, P["kind"], P["p"], P["r"], P["pw"],
                             np.array([2], dtype=np.int64))
        g_bf = (S[0, 0] + 1j * S[0, 1]) / np.sqrt(int(norms[t]))
        worst = max(worst, abs(g_bf - g_crt[t]))
    print(f"CRT formula vs brute force for {len(norms) - 1} squarefree n (N <= {Xc}): max error {worst:.2e}")
    return worst < 1e-8


def main(X):
    t0 = time.time()
    P = prime_ideals(X)
    gens = gauss.primary_generators(P)
    gam_p = prime_gamma2(P, gens)
    print(f"gamma_2 at {len(gam_p)} primes up to {X:,}: {time.time() - t0:.0f} s", flush=True)
    norms, mu, ptr, idx = squarefree_ideals(P, 1, X)
    g = composite_gamma2(P, gens, gam_p, ptr, idx)
    print(f"gamma_2 at {len(norms):,} squarefree n: {time.time() - t0:.0f} s", flush=True)
    np.savez(f"runs/patterson_X{X}.npz", norms=norms, mu=mu, g=g,
             pnorms=P["norm"], gp=gam_p)


if __name__ == "__main__":
    if check_crt():
        main(int(sys.argv[1]))
