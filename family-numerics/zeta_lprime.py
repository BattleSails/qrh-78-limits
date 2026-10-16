"""
Control for lowzeros.py: |zeta'(rho)| / log(gamma / 2 pi) at zeros n = 1500..4520 (heights
~2000-5000, Keating-Snaith size N = log(gamma/2 pi) ~ 6) from the LMFDB cache, compared with
the same statistic for CUE N = 6.

  python zeta_lprime.py
"""
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np

EPS = np.array([0.05, 0.1, 0.2, 0.4])


def lp(g):
    mp.mp.dps = 20
    return float(abs(mp.zeta(mp.mpc(0.5, g), derivative=1)))


if __name__ == "__main__":
    g = np.load(r"T:\_Data\RiemannZeros\npz\zeros_14.npz")["gamma"][1500:]
    with ProcessPoolExecutor(16) as ex:
        v = np.array(list(ex.map(lp, g, chunksize=50)))
    x = v / np.log(g / (2 * np.pi))
    w = x / np.median(x)
    print(f"zeta, {len(g)} zeros at heights {g[0]:.0f}-{g[-1]:.0f}: Pr(|zeta'|/median < eps) = "
          + " ".join(f"{np.mean(w < e):.2e}" for e in EPS) + f";  std log = {np.std(np.log(w)):.3f}")
    np.save("runs/zeta_lprime.npy", np.c_[g, v])
