"""Checks of the Z[w] arithmetic against known identities. Run: python test_eisenstein.py"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import (prime_ideals, sextic_exponent, squarefree_ideals, elements_upto)

P = prime_ideals(3000)
nP = len(P["norm"])
fails = 0


def chi(u, i):
    return sextic_exponent(u[0], u[1], P["kind"][i], P["p"][i], P["r"][i], P["pw"][i])


def check(name, ok):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    fails += not ok


def mul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0] - a[1] * b[1])


def norm(a):
    return a[0] ** 2 - a[0] * a[1] + a[1] ** 2


# 1. multiplicativity in u and values in 0..5 / -1
rng = np.random.default_rng(1)
ok = True
for _ in range(3000):
    i = rng.integers(nP)
    u = tuple(int(v) for v in rng.integers(-60, 60, 2))
    v = tuple(int(v) for v in rng.integers(-60, 60, 2))
    a, b, c = chi(u, i), chi(v, i), chi(mul(u, v), i)
    if a < 0 or b < 0:
        ok &= (c == -1)
    else:
        ok &= (c == (a + b) % 6)
check("multiplicative in u: chi(uv) = chi(u) chi(v)", ok)

# 2. character sum over a complete residue system vanishes, and all six values occur
ok = True
for i in range(nP):
    if P["norm"][i] > 400:
        break
    p, kind = int(P["p"][i]), P["kind"][i]
    vals = []
    if kind == 0:
        vals = [chi((a, 0), i) for a in range(p)]       # x + 0 w covers F_p
    else:
        vals = [chi((a, b), i) for a in range(p) for b in range(p)]
    nz = [v for v in vals if v >= 0]
    s = sum(np.exp(1j * np.pi * v / 3) for v in nz)
    ok &= abs(s) < 1e-8 and len(set(nz)) == 6 and vals.count(-1) == 1
check("sum of (u/P)_6 over O/P is 0, all six values occur, one zero", ok)

# 3. quadratic part: for split P over p and rational a, (a/P)_2 = Legendre(a, p);
#    for inert (q), rational a are squares in F_{q^2}: (a/q)_2 = 1
ok = True
for i in range(nP):
    p = int(P["p"][i])
    for a in range(1, 40):
        if a % p == 0:
            continue
        j = chi((a, 0), i)
        quad = 1 if j % 2 == 0 else -1                  # chi^3 = (-1)^j
        want = pow(a, (p - 1) // 2, p) if P["kind"][i] == 0 else 1
        want = 1 if want == 1 else -1
        ok &= quad == want
check("quadratic symbol: (a/P)_2 = Legendre(a,p) for split P, 1 for inert", ok)

# 4. cubic part: rational a is a cube mod p  <=>  (a/P)_3 = 1 for split P
ok = True
for i in range(nP):
    if P["kind"][i] != 0:
        continue
    p = int(P["p"][i])
    for a in range(1, 40):
        if a % p == 0:
            continue
        is_cube = pow(a, (p - 1) // 3, p) == 1
        ok &= is_cube == (chi((a, 0), i) % 3 == 0)      # chi^2 trivial <=> j = 0 mod 3
check("cubic symbol: a is a cube mod p iff (a/P)_3 = 1", ok)


# 5. cubic reciprocity for primary primes (= 1 mod 3): (pi1/pi2)_3 = (pi2/pi1)_3
def primary_generator(i):
    p, r = int(P["p"][i]), int(P["r"][i])
    if P["kind"][i] == 1:
        return (-p, 0)                                   # -q = 1 mod 3
    for a, b in itertools.product(range(-60, 61), repeat=2):
        if norm((a, b)) == p and (a + b * r) % p == 0:
            g = (a, b)
            break
    units = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]
    for e in units:
        h = mul(g, e)
        if h[0] % 3 == 1 and h[1] % 3 == 0:
            return h
    raise ValueError(i)


gens = [primary_generator(i) for i in range(min(nP, 60))]
ok = True
pairs = 0
for i, j in itertools.combinations(range(len(gens)), 2):
    if P["p"][i] == P["p"][j]:
        continue
    a, b = chi(gens[i], j), chi(gens[j], i)
    ok &= (2 * a) % 6 == (2 * b) % 6                    # cubic symbol = chi^2
    pairs += 1
check(f"cubic reciprocity on {pairs} pairs of primary primes", ok)

# 6. ideal count: number of ideals coprime to 6 of norm <= x matches a direct count
x = 3000
norms, mu, ptr, idx = squarefree_ideals(P, 1, x)
xs, ys, nm = elements_upto(x)
# elements coprime to 6 counted up to units (6 units) give all ideals coprime to 6;
# keep squarefree ones by checking divisibility by squares of prime ideals
cop = [(a, b) for a, b, n in zip(xs, ys, nm) if n % 2 != 0 and n % 3 != 0]
sqf = 0
for u in cop:
    good = True
    for i in range(nP):
        if P["norm"][i] ** 2 > norm(u):
            break
        if chi(u, i) == -1:
            # P | u; check P^2 | u via u / gen(P)
            g = primary_generator(i) if i < 60 else None
            if g is None:
                continue
            # u * conj(g) / N(g) must still be divisible by P
            gc = (g[0] - g[1], -g[1])
            t = mul(u, gc)
            Ng = norm(g)
            if t[0] % Ng == 0 and t[1] % Ng == 0 and chi((t[0] // Ng, t[1] // Ng), i) == -1:
                good = False
                break
    sqf += good
check(f"squarefree ideals of norm <= {x}: enumeration {len(norms)} vs element count {sqf // 6}",
      len(norms) == sqf // 6)

print("\nALL TESTS PASSED" if fails == 0 else f"\n{fails} TEST(S) FAILED")
