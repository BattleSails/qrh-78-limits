"""
Summaries of the family runs.

For each run and row range H = D^kappa:
  R(kappa) = sum_{N(u) <= H} |A_u|^2 / sum_{N(u) <= H} mass_u
           (1 = square-root cancellation, random-model normalisation)
plus the largest normalised rows and the size of A_1 relative to sqrt(mass_1).

  python analyze.py runs/D*.npz
"""
import sys

import numpy as np

KAPPAS = [0.5, 0.6, 0.7, 6 / 7, 0.95, 1.0, 1.05, 1.1]


def summarize(path):
    d = np.load(path, allow_pickle=False)
    D, un, A, mass = int(d["D"]), d["un"], d["A"], d["mass"]
    mode = str(d["mode"]) if "mode" in d.files else "mobius"
    a2 = np.abs(A) ** 2
    row = {"path": path, "D": D, "mode": mode}
    for k in KAPPAS:
        m = un <= D**k
        if m.sum() and D**k <= un.max() * 1.0001 + 1:
            row[k] = a2[m].sum() / mass[m].sum()
    one = (d["ux"] == 1) & (d["uy"] == 0)
    row["A1_ratio"] = float(abs(A[one][0]) / np.sqrt(mass[one][0]))
    z = a2 / np.where(mass > 0, mass, np.nan)
    order = np.argsort(-np.nan_to_num(z))[:5]
    row["top"] = [(int(d["ux"][i]), int(d["uy"][i]), int(un[i]), float(z[i])) for i in order]
    row["nrows"] = len(un)
    return row


if __name__ == "__main__":
    rows = [summarize(p) for p in sys.argv[1:]]
    rows.sort(key=lambda r: (r["mode"], r["D"]))
    hdr = " ".join(f"k={k:.3f}" for k in KAPPAS)
    print(f"{'mode':>7} {'D':>7} {'rows':>8} | {hdr} | |A_1|/sqrt(mass)")
    for r in rows:
        vals = " ".join(f"{r[k]:7.4f}" if k in r else "      -" for k in KAPPAS)
        print(f"{r['mode']:>7} {r['D']:7d} {r['nrows']:8d} | {vals} | {r['A1_ratio']:.3f}")
    print("\nLargest rows |A_u|^2 / mass_u  (u = x + y w, N(u)):")
    for r in rows:
        print(f"  {r['mode']:>7} D={r['D']:6d}: " +
              ", ".join(f"({x},{y}) N={n} -> {v:.1f}" for x, y, n, v in r["top"][:3]))
