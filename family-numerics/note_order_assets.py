"""LaTeX table for the lower-order families (reads runs/D*.npz, runs/ctl_plain_*, runs/ord*)."""
import numpy as np

KS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BS = "\\"
rows = []
for m, k, name in [(1, 6, "sextic"), (2, 3, "cubic"), (3, 2, "quadratic")]:
    for mode in ["mobius", "plain"]:
        for D in [20000, 80000]:
            if m == 1:
                f = f"runs/D{D}.npz" if mode == "mobius" else f"runs/ctl_plain_D{D}.npz"
            else:
                f = f"runs/ord{m}{'' if mode == 'mobius' else 'plain'}_D{D}.npz"
            d = np.load(f)
            un, a2, ms = d["un"], np.abs(d["A"]) ** 2, d["mass"]
            R = [a2[un <= D**x].sum() / ms[un <= D**x].sum() for x in KS]
            s = un <= D
            m4 = np.mean(a2[s] ** 2) / np.mean(a2[s]) ** 2
            label = f"{name} ($k={k}$), {'M{\\\"o}bius' if mode == 'mobius' else 'plain'}"
            fmt = (lambda v: f"{v:.3f}") if mode == "mobius" else (lambda v: f"{v:.1f}")
            rows.append(f"{label} & {D:,} & " + " & ".join(fmt(v) for v in R) +
                        (f" & {m4:.2f}" if mode == "mobius" else " & --") + " " + BS * 2)
    rows.append(BS + "midrule")
open("note/tab_order.tex", "w").write("\n".join(rows[:-1]) + "\n")
print(open("note/tab_order.tex").read())
