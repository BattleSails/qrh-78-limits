"""
Spectral mass over the family: is MS(kappa) the family average of
  M_u = sum_rho |What(rho) / L^S'(rho, psi_u)|^2 ?

For every row u with 1 < N(u) <= HMAX, rowchar.gp (PARI/GP) gives the zeros of L(s, psi_u)
with |gamma| <= T and their residues c_rho; M_u = sum |c_rho|^2 (the tail beyond height 50 is
~1e-4 of the total, checked on three rows at height 100). The measured side is the family code
itself: |A_u(D)|^2 / D and the random-model mass / D at several D.

  python famspec.py HMAX T
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eisenstein import elements_upto, prime_ideals, squarefree_ideals
from family import W, family_sums
from rowspec import GP

DS = [10_000, 100_000, 1_000_000]


def pari_row(x, y, T):
    fn = f"runs/fam/rowspec_{x}_{y}.txt"
    if not os.path.exists(fn):
        tmp = f"runs/rowspec_{x}_{y}.txt"
        if os.path.exists(tmp):
            os.remove(tmp)
        r = subprocess.run([GP, "-q", "-s", "2G"], input=f'read("rowchar.gp"); rowspec({x}, {y}, {T})',
                           capture_output=True, text=True)
        if not os.path.exists(tmp):
            return x, y, None, r.stdout[-300:] + r.stderr[-300:]
        os.replace(tmp, fn)
        info = [l.strip() for l in r.stdout.splitlines() if l.startswith("u =")]
        with open(fn + ".info", "w") as f:
            f.write("\n".join(info))
    with open(fn) as f:
        d = np.array([[float(v) for v in l.replace(" e", "e").split()] for l in f if l.strip()])
    s0, c = d[:, 0] + 1j * d[:, 1], d[:, 2] + 1j * d[:, 3]
    on = np.abs(s0.real - 0.5) < 1e-9
    info = open(fn + ".info").read() if os.path.exists(fn + ".info") else ""
    return x, y, (np.sum(np.abs(c[on]) ** 2), int(on.sum()), info), ""


def main(hmax, T):
    os.makedirs("runs/fam", exist_ok=True)
    ux, uy, un = elements_upto(hmax)
    keep = un > 1
    ux, uy, un = ux[keep], uy[keep], un[keep]
    print(f"{len(ux)} rows with 1 < N(u) <= {hmax}", flush=True)
    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(16) as ex:
        for x, y, r, err in ex.map(lambda a: pari_row(int(a[0]), int(a[1]), T), zip(ux, uy)):
            if r is None:
                print(f"  u = ({x},{y}) FAILED: {err}", flush=True)
            res[(x, y)] = r
    print(f"PARI side done in {time.time() - t0:.0f} s", flush=True)
    ok = np.array([res[(int(x), int(y))] is not None for x, y in zip(ux, uy)])
    Mspec = np.array([res[(int(x), int(y))][0] if o else np.nan for x, y, o in zip(ux, uy, ok)])
    A2 = np.zeros((len(DS), len(ux)))
    mass = np.zeros((len(DS), len(ux)))
    for i, D in enumerate(DS):
        P = prime_ideals(2 * D)
        norms, mu, ptr, idx = squarefree_ideals(P, D, 2 * D)
        Are, Aim, ms = family_sums(ux, uy, P["kind"], P["p"], P["r"], P["pw"], mu * W(norms / D), ptr, idx)
        A2[i] = (Are ** 2 + Aim ** 2) / D
        mass[i] = ms / D
    np.savez("runs/famspec.npz", ux=ux, uy=uy, un=un, Mspec=Mspec, A2=A2, mass=mass, Ds=DS)
    m = ok
    print(f"\nrows used: {m.sum()}")
    print(f"family mean of spectral mass M_u          : {Mspec[m].mean():.4f}")
    for i, D in enumerate(DS):
        print(f"D = {D:>9,}: mean |A_u|^2/D = {A2[i, m].mean():.4f}, mean mass/D = {mass[i, m].mean():.4f}, "
              f"R = {A2[i, m].sum() / mass[i, m].sum():.3f}")
    A2bar = A2[:, m].mean(0)
    print(f"per-row correlation of M_u with |A_u|^2/D averaged over the {len(DS)} D's: "
          f"{np.corrcoef(Mspec[m], A2bar)[0, 1]:.3f}")
    o = np.argsort(-Mspec[m])
    print("largest M_u:")
    for j in o[:12]:
        print(f"  u = ({ux[m][j]:3d},{uy[m][j]:3d}) N = {un[m][j]:3d}: M_u = {Mspec[m][j]:.3f}, "
              f"mean |A|^2/D = {A2bar[j]:.3f}, mass/D = {mass[1, m][j]:.3f}")
    q = np.quantile(Mspec[m], [0.1, 0.25, 0.5, 0.75, 0.9])
    print("quantiles of M_u (10/25/50/75/90%): " + " ".join(f"{v:.3f}" for v in q))


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]))
