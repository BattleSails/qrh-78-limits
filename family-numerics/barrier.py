"""
The D + H barrier made concrete (item 7 of the caveat plan).

M_{u,n} = chi_n(u) for rows u in Z[w], 0 < N(u) <= H, and columns the squarefree ideals
n prime to 6 with D <= N n <= 2D. For general coefficients the best mean-square constant is
Delta = ||M||^2 = lambda_max(M^H M); trivially Delta >= max(#rows, #cols).
We compute Delta exactly (Gram matrix accumulated over rows sorted by norm) at several row
ranges H = D^kappa, with and without the rows that duplicate other rows (u divisible by
lambda^6 or 2^6, where lambda = 1 + 2w), and compare with the Rayleigh quotient of the Mobius
vector mu(n) W(N n/D).

  python barrier.py D
"""
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import prime_ideals, sextic_exponent, squarefree_ideals, elements_upto
from family import W

KAPPAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@njit(parallel=True, cache=True)
def exponent_matrix(ux, uy, kind, p, r, pw, ptr, idx):
    nU, nP, nN = len(ux), len(p), len(ptr) - 1
    E = np.empty((nU, nN), dtype=np.int8)
    for k in prange(nU):
        e = np.empty(nP, dtype=np.int64)
        for i in range(nP):
            e[i] = sextic_exponent(ux[k], uy[k], kind[i], p[i], r[i], pw[i])
        for n in range(nN):
            s = 0
            zero = False
            for t in range(ptr[n], ptr[n + 1]):
                v = e[idx[t]]
                if v < 0:
                    zero = True
                    break
                s += v
            E[k, n] = -1 if zero else s % 6
    return E


def duplicate_rows(ux, uy):
    """u divisible by lambda^6 = -27 or by 2^6 = 64 (as elements of Z[w])."""
    def divisible(x, y, m):
        # (x + y w)/m in Z[w]  <=>  m | x and m | y   for rational m
        return (x % m == 0) & (y % m == 0)
    return divisible(ux, uy, 27) | divisible(ux, uy, 64)


def run(D):
    t0 = time.time()
    P = prime_ideals(2 * D)
    norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
    a = (mu * W(norms / D)).astype(np.complex128)
    ux, uy, un = elements_upto(int(D ** max(KAPPAS)) + 1)
    E = exponent_matrix(ux, uy, P["kind"], P["p"], P["r"], P["pw"], ptr, idx)
    z6 = np.exp(1j * np.pi * np.arange(6) / 3)
    dup = duplicate_rows(ux, uy)
    nN = len(norms)
    print(f"D={D}: {nN} columns, {len(ux)} rows built in {time.time() - t0:.0f} s", flush=True)
    print(f"{'kappa':>6} {'rows':>7} {'cols':>6} | {'Delta all':>10} {'Delta/max':>9} | "
          f"{'Delta no-dup':>12} {'/max':>6} | {'Mobius RQ':>9} {'RQ/rows':>7} | {'Delta/Mobius':>12}")
    for dedup in (False,):
        pass
    G_all = np.zeros((nN, nN), dtype=np.complex128)
    G_nd = np.zeros((nN, nN), dtype=np.complex128)
    start = 0
    for kappa in KAPPAS:
        H = D ** kappa
        stop = np.searchsorted(un, H, side="right")
        for c0 in range(start, stop, 4000):
            c1 = min(stop, c0 + 4000)
            Eb = E[c0:c1]
            Mb = np.where(Eb >= 0, z6[np.maximum(Eb, 0)], 0)
            G_all += Mb.conj().T @ Mb
            keep = ~dup[c0:c1]
            G_nd += Mb[keep].conj().T @ Mb[keep]
        start = stop
        lam_all = np.linalg.eigvalsh(G_all)[-1]
        lam_nd = np.linalg.eigvalsh(G_nd)[-1]
        rows, rows_nd = stop, int(np.sum(~dup[:stop]))
        rq = np.real(a.conj() @ (G_all @ a)) / np.real(a.conj() @ a)
        print(f"{kappa:6.2f} {rows:7d} {nN:6d} | {lam_all:10.1f} {lam_all / max(rows, nN):9.3f} | "
              f"{lam_nd:12.1f} {lam_nd / max(rows_nd, nN):6.3f} | {rq:9.1f} {rq / rows:7.3f} | "
              f"{lam_all / rq:12.2f}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]))
