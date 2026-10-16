"""
Arithmetic of O = Z[w], w = exp(2 pi i/3), for the family experiment.

Elements are pairs (x, y) meaning x + y w;  N(x + y w) = x^2 - x y + y^2.

Prime ideals coprime to 6:
  split  p = 1 mod 3: two ideals of norm p, each the kernel of a ring map
         O -> F_p, w -> r, with r a primitive cube root of unity mod p
         (the two ideals use r and r^2 = -1 - r);
  inert  q = 2 mod 3, q >= 5: one ideal (q) of norm q^2, residue field
         F_q[w] / (w^2 + w + 1).

Sextic residue symbol (u / P)_6 for P not dividing u: the unique sixth root of
unity zeta in O with zeta = u^{(N P - 1)/6} mod P. We return it as an exponent
j in 0..5 meaning exp(i pi j / 3), using that zeta_6 = exp(i pi/3) = 1 + w
(since -w^2 = 1 + w). The value -1 means P divides u (symbol 0).
"""
import numpy as np
from numba import njit


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def cube_root_of_unity(p):
    for a in range(2, p):
        r = pow(a, (p - 1) // 3, p)
        if r != 1:
            return r
    raise ValueError(p)


def prime_ideals(max_norm):
    """Prime ideals of O coprime to 6 with norm <= max_norm, sorted by norm.

    Returns a dict of numpy arrays:
      kind  0 split / 1 inert,  p  rational prime,  r  cube root used (split),
      norm, pw  (6 powers of zeta_6 mod p; split only).
    """
    rows = []
    for p in primes_upto(max_norm):
        p = int(p)
        if p % 3 == 1:
            r = cube_root_of_unity(p)
            for rr in (r, (-1 - r) % p):
                rows.append((p, 0, p, rr))
        elif p % 3 == 2 and p >= 5 and p * p <= max_norm:
            rows.append((p * p, 1, p, 0))
    rows.sort()
    norm = np.array([x[0] for x in rows], dtype=np.int64)
    kind = np.array([x[1] for x in rows], dtype=np.int64)
    p = np.array([x[2] for x in rows], dtype=np.int64)
    r = np.array([x[3] for x in rows], dtype=np.int64)
    pw = np.zeros((len(rows), 6), dtype=np.int64)
    for i in range(len(rows)):
        if kind[i] == 0:
            z = (1 + r[i]) % p[i]              # image of zeta_6 = 1 + w
            v = 1
            for j in range(6):
                pw[i, j] = v
                v = v * z % p[i]
    return dict(norm=norm, kind=kind, p=p, r=r, pw=pw)


@njit(cache=True)
def _powmod(a, e, m):
    res = 1
    a %= m
    while e > 0:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res


@njit(cache=True)
def _fq2_mul(a, b, c, d, q):
    # (a + b w)(c + d w) = ac - bd + (ad + bc - bd) w   using w^2 = -1 - w
    return (a * c - b * d) % q, (a * d + b * c - b * d) % q


@njit(cache=True)
def _fq2_pow(a, b, e, q):
    ra, rb = 1, 0
    a %= q
    b %= q
    while e > 0:
        if e & 1:
            ra, rb = _fq2_mul(ra, rb, a, b, q)
        a, b = _fq2_mul(a, b, a, b, q)
        e >>= 1
    return ra, rb


@njit(cache=True)
def sextic_exponent(x, y, kind, p, r, pw):
    """Exponent j of (x + y w / P)_6 = exp(i pi j/3); -1 if P | u."""
    if kind == 0:
        a = (x + y * r) % p
        if a == 0:
            return -1
        b = _powmod(a, (p - 1) // 6, p)
        for j in range(6):
            if pw[j] == b:
                return j
        return -2                                  # cannot happen
    q = p
    if x % q == 0 and y % q == 0:
        return -1
    ba, bb = _fq2_pow(x, y, (q * q - 1) // 6, q)
    za, zb = 1, 0                                  # powers of zeta_6 = 1 + w
    for j in range(6):
        if za == ba and zb == bb:
            return j
        za, zb = _fq2_mul(za, zb, 1, 1, q)
    return -2


def squarefree_ideals(P, lo, hi):
    """Squarefree products of distinct prime ideals in P with lo <= norm <= hi.

    Returns (norms, mu, factor_ptr, factor_idx) in CSR form.
    """
    norms_p = P["norm"]
    out_norm, out_fac = [], []
    stack = [(0, 1, ())]
    while stack:
        start, nrm, fac = stack.pop()
        if nrm >= lo:
            out_norm.append(nrm)
            out_fac.append(fac)
        for i in range(start, len(norms_p)):
            m = nrm * int(norms_p[i])
            if m > hi:
                break
            stack.append((i + 1, m, fac + (i,)))
    order = np.argsort(out_norm, kind="stable")
    norms = np.array(out_norm, dtype=np.int64)[order]
    facs = [out_fac[i] for i in order]
    ptr = np.zeros(len(facs) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(f) for f in facs])
    idx = np.array([i for f in facs for i in f], dtype=np.int64)
    mu = np.array([(-1) ** len(f) for f in facs], dtype=np.int64)
    return norms, mu, ptr, idx


def elements_upto(H):
    """All nonzero x + y w with norm <= H, as arrays (x, y, norm)."""
    R = int(np.ceil(2 * np.sqrt(H / 3))) + 2       # |y| <= 2 sqrt(H/3)
    xs, ys = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    nm = xs * xs - xs * ys + ys * ys
    keep = (nm > 0) & (nm <= H)
    o = np.argsort(nm[keep], kind="stable")
    return xs[keep][o].astype(np.int64), ys[keep][o].astype(np.int64), nm[keep][o].astype(np.int64)
