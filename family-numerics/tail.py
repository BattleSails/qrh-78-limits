"""
Tail of the family rows, split by row norm N(u).

Rows u and conj(u) carry conjugate values (mu is conjugation-symmetric), so we keep
one row per conjugate pair. For each norm bin we report the 4th-moment ratio
E z^2 / (E z)^2 with z = |A_u|^2 / mass_u (complex Gaussian: 2), its jackknife
error over 20 random row groups, and the excess tail fraction P(z > 8) / e^-8.

  python tail.py runs/D80000.npz runs/ctl_random_D80000.npz ...
"""
import sys

import numpy as np

EDGES = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def one_per_conjugate_pair(x, y):
    # conj(x + y w) = (x - y) - y w ; keep the lexicographically larger representative
    cx, cy = x - y, -y
    return (x > cx) | ((x == cx) & (y >= cy))


def ratio_and_err(z, rng):
    k = np.mean(z**2) / np.mean(z) ** 2
    g = rng.integers(0, 20, len(z))
    jk = []
    for i in range(20):
        zz = z[g != i]
        jk.append(np.mean(zz**2) / np.mean(zz) ** 2)
    jk = np.array(jk)
    return k, np.sqrt(19 / 20 * np.sum((jk - jk.mean()) ** 2))


def report(path):
    d = np.load(path, allow_pickle=False)
    mode = str(d["mode"]) if "mode" in d.files else "mobius"
    keep = one_per_conjugate_pair(d["ux"], d["uy"])
    un = d["un"][keep]
    z = (np.abs(d["A"]) ** 2 / d["mass"])[keep]
    rng = np.random.default_rng(0)
    cells = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (un >= lo) & (un < hi)
        if m.sum() < 200:
            cells.append(" " * 27)
            continue
        k, e = ratio_and_err(z[m], rng)
        tail = np.mean(z[m] > 8) / np.exp(-8)
        cells.append(f"{k:5.3f}+/-{e:5.3f} t={tail:5.1f} n={m.sum():<6d}"[:27].ljust(27))
    print(f"{mode:>6} D={int(d['D']):6d} | " + " | ".join(cells))


if __name__ == "__main__":
    head = " | ".join(f"N(u) in [{lo:.0e},{hi:.0e})".ljust(27) for lo, hi in zip(EDGES[:-1], EDGES[1:]))
    print(f"{'':>15} | {head}")
    print("   (each cell: 4th-moment ratio +/- jackknife error, t = P(z>8)/e^-8, n = rows)")
    for p in sys.argv[1:]:
        report(p)
