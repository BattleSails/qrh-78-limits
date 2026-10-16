"""
Gauss sums over Z[w] and a brute-force check of Lemma 3.2 (eq:gj--eq:convert2) of
OpenAI, "The Quasi-Riemann Hypothesis" (alternate 11/12 proof, Oct 2026).

For squarefree primary n = a + b w (n = 1 mod 3), coprime to 6:
  gamma_j(n) = N(n)^{-1/2} sum_{x mod n} chi_n(x)^j e(x/n),
  e(z) = exp(4 pi i Im z / sqrt 3),  chi_n(x) = (x/n)_6.
With x * conj(n) = u + v w we have Im(x/n) = (sqrt3/2) v / N(n), so e(x/n) = exp(2 pi i v/N(n)).
A complete residue system mod n is {x + y w : 0 <= x < N/g, 0 <= y < g}, g = gcd(a, b).

Nothing here uses multiplicativity or reciprocity: every Gauss sum is a full sum.
"""
import os
import sys
from math import gcd

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals


def mul(u, v):
    return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0] - u[1] * v[1])


def primary_generators(P):
    """Primary generator (a, b) (a = 1, b = 0 mod 3) of each prime ideal in P."""
    gens = []
    units = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]
    for i in range(len(P["norm"])):
        p, r = int(P["p"][i]), int(P["r"][i])
        if P["kind"][i] == 1:
            gens.append((-p, 0))                         # -q = 1 mod 3
            continue
        g = None
        bmax = int(2 * np.sqrt(p / 3)) + 2
        for b in range(-bmax, bmax + 1):
            # a^2 - a b + b^2 - p = 0
            disc = 4 * p - 3 * b * b
            if disc < 0:
                continue
            s = int(round(np.sqrt(disc)))
            for ss in (s - 1, s, s + 1):
                if ss >= 0 and ss * ss == disc:
                    for a2 in (b + ss, b - ss):
                        if a2 % 2 == 0:
                            a = a2 // 2
                            if (a + b * r) % p == 0:
                                g = (a, b)
            if g:
                break
        for e in units:
            h = mul(g, e)
            if h[0] % 3 == 1 and h[1] % 3 == 0:
                gens.append(h)
                break
        else:
            raise ValueError(i)
    return gens


@njit(cache=True)
def gauss_sums(a, b, N, fac, kind, p, r, pw, js):
    """sum_{x mod n} chi_n(x)^j e(x/n) for each j in js (unnormalised)."""
    g = 1
    aa, bb = abs(a), abs(b)
    while bb:
        aa, bb = bb, aa % bb
    g = aa
    d1 = N // g
    ca, cb = a - b, -b                                  # conj(n)
    re6 = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    im6 = np.array([0.0, 0.8660254037844386, 0.8660254037844386, 0.0,
                    -0.8660254037844386, -0.8660254037844386])
    out = np.zeros((len(js), 2))
    for y in range(g):
        for x in range(d1):
            s = 0
            zero = False
            for t in range(len(fac)):
                i = fac[t]
                v = sextic_exponent(x, y, kind[i], p[i], r[i], pw[i])
                if v < 0:
                    zero = True
                    break
                s += v
            if zero:
                continue
            # x*conj(n) = (x + y w)(ca + cb w): w-coefficient
            vco = (x * cb + y * ca - y * cb) % N
            ang = 2.0 * np.pi * vco / N
            c, si = np.cos(ang), np.sin(ang)
            for k in range(len(js)):
                ee = (js[k] * s) % 6
                out[k, 0] += re6[ee] * c - im6[ee] * si
                out[k, 1] += re6[ee] * si + im6[ee] * c
    return out


def check_lemma(X):
    P = prime_ideals(X)
    gens = primary_generators(P)
    norms, mu, ptr, idx = squarefree_ideals(P, 1, X)
    js = np.array([1, 2, 3, -1], dtype=np.int64)
    w = complex(-0.5, np.sqrt(3) / 2)
    z6 = lambda e: np.exp(1j * np.pi * e / 3)
    worst = {k: 0.0 for k in ["|g_j|=1", "eq:gj a", "eq:gj b", "eq:g", "convert2", "gamma3 table"]}
    count = 0
    for t in range(1, len(norms)):                       # skip the unit ideal
        fac = idx[ptr[t]:ptr[t + 1]]
        n = (1, 0)
        for i in fac:
            n = mul(n, gens[i])
        a, b = n
        assert a % 3 == 1 and b % 3 == 0 and a * a - a * b + b * b == norms[t]
        N = int(norms[t])
        S = gauss_sums(a, b, N, fac, P["kind"], P["p"], P["r"], P["pw"], js)
        g1, g2, g3, gm1 = [(S[k, 0] + 1j * S[k, 1]) / np.sqrt(N) for k in range(4)]
        chi = lambda x, y: sum(sextic_exponent(x, y, P["kind"][i], P["p"][i], P["r"][i], P["pw"][i])
                               for i in fac)
        chi_m1 = z6(chi(-1, 0) % 6)
        chi_4 = z6(chi(4, 0) % 6)
        alpha = (a + b * w) / abs(a + b * w)
        m = int(mu[t])
        G = np.conj(chi_4) * g3
        quad = (1 + 1j ** (-b) + 1j ** a + 1j ** (b - a)) / 2   # Gamma_quad(a + b w)
        worst["|g_j|=1"] = max(worst["|g_j|=1"], max(abs(abs(g) - 1) for g in (g1, g2, g3, gm1)))
        worst["eq:gj a"] = max(worst["eq:gj a"], abs(g2 ** 3 - m * alpha))
        worst["eq:gj b"] = max(worst["eq:gj b"], abs(g1 * g2 - m * alpha * G))
        worst["eq:g"] = max(worst["eq:g"], abs(g1 * gm1 - chi_m1))
        worst["convert2"] = max(worst["convert2"], abs(m * gm1 - chi_m1 / G * np.conj(alpha) * g2))
        worst["gamma3 table"] = max(worst["gamma3 table"], abs(g3 - quad))
        count += 1
    return count, worst


if __name__ == "__main__":
    X = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    count, worst = check_lemma(X)
    print(f"{count} squarefree primary n coprime to 6 with N(n) <= {X}")
    for k, v in worst.items():
        print(f"  [{'PASS' if v < 1e-8 else 'FAIL'}] {k:13s} max error {v:.2e}")
