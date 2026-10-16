"""Figure and LaTeX tables for the note (reads runs/*.npz)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy import predicted_ratio

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "savefig.bbox": "tight"})
os.makedirs("note", exist_ok=True)
DS = [2500, 5000, 10000, 20000, 40000, 80000, 160000, 320000]
CTL = [5000, 20000, 80000, 160000, 320000]
KAP = np.linspace(0.5, 1.1, 25)


def load(D, mode="mobius"):
    f = f"runs/D{D}.npz" if mode == "mobius" else f"runs/ctl_{mode}_D{D}.npz"
    return np.load(f) if os.path.exists(f) else None


def R_curve(d):
    D, un, a2, m = int(d["D"]), d["un"], np.abs(d["A"]) ** 2, d["mass"]
    out = []
    for k in KAP:
        sel = un <= D**k
        out.append(a2[sel].sum() / m[sel].sum() if D**k <= un.max() * 1.001 + 2 and sel.sum() > 50 else np.nan)
    return np.array(out)


def raw_ratio(d):
    a2 = np.abs(d["A"]) ** 2
    return np.mean(a2**2) / np.mean(a2) ** 2


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axs[0]
ctl_curves = np.array([R_curve(load(D, "random")) for D in CTL])
lo, hi = np.nanmin(ctl_curves, axis=0), np.nanmax(ctl_curves, axis=0)
ax.fill_between(KAP, lo, hi, color=GRAY, alpha=0.25, lw=0, label="random signs (range over 5 runs)")
for D, col in [(20000, BLUE), (80000, ORANGE), (320000, AQUA)]:
    ax.plot(KAP, R_curve(load(D)), color=col, marker="o", ms=2.5, label=f"Möbius, $D={D:,}$")
ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.axvline(6 / 7, color=INK2, lw=0.8, ls=(0, (1, 2)))
ax.text(1.005, 1.085, "$H=D$", color=INK2, fontsize=7.5)
ax.text(6 / 7 + 0.005, 1.085, "$H=D^{6/7}$", color=INK2, fontsize=7.5)
ax.set_ylim(0.80, 1.10)
ax.set_xlabel(r"row range exponent $\kappa$  ($H=D^\kappa$)")
ax.set_ylabel(r"$R=\sum|A_u|^2/\sum\mathrm{mass}_u$")
ax.legend(fontsize=7, loc="lower right")
ax.set_title("(a)", loc="left", fontsize=9)

ax = axs[1]
mob = [(D, raw_ratio(load(D))) for D in DS if load(D) is not None]
rnd = [(D, raw_ratio(load(D, "random"))) for D in CTL]
pred = [(D, predicted_ratio(D)[0]) for D in [2500, 5000, 10000, 20000]]
ax.axhline(2.0, color=GRAY, lw=1.2, ls=(0, (4, 3)))
ax.text(2600, 2.005, "complex Gaussian", color=INK2, fontsize=7.5, va="bottom")
ax.plot(*zip(*mob), "o-", color=BLUE, ms=3.5, label="Möbius family (measured)")
ax.plot(*zip(*pred), "s", color=ORANGE, ms=5, mfc="none", mew=1.4, label="multiplicative-energy prediction")
ax.plot(*zip(*rnd), "^-", color=GRAY, ms=3.5, label="random signs (measured)")
ax.set_xscale("log")
ax.set_ylim(1.97, 2.22)
ax.set_xlabel("column scale $D$")
ax.set_ylabel(r"$\mathbb{E}|A_u|^4/(\mathbb{E}|A_u|^2)^2$")
ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.22))
ax.set_title("(b)", loc="left", fontsize=9)
fig.savefig("note/fig_family.pdf")

# Table 1: R(kappa) at selected kappa
BS = "\\"
ks = [0.5, 0.6, 0.7, 6 / 7, 1.0, 1.1]
with open("note/tab_R.tex", "w") as f:
    for D in DS:
        d = load(D)
        un, a2, m = d["un"], np.abs(d["A"]) ** 2, d["mass"]
        cells = []
        for k in ks:
            sel = un <= D**k
            ok = D**k <= un.max() * 1.001 + 2
            cells.append(f"{a2[sel].sum() / m[sel].sum():.3f}" if ok else "--")
        r = load(D, "random")
        rc = "--"
        if r is not None:
            sel = r["un"] <= D ** (6 / 7)
            rc = f"{(np.abs(r['A'][sel])**2).sum() / r['mass'][sel].sum():.3f}"
        f.write(f"{D:,} & {len(un):,} & " + " & ".join(cells) + f" & {rc} " + BS * 2 + "\n")

# Table 2: fourth moments
with open("note/tab_m4.tex", "w") as f:
    pd = dict(pred)
    rd = dict(rnd)
    for D, v in mob:
        p = f"{pd[D]:.3f}" if D in pd else "--"
        q = f"{rd[D]:.3f}" if D in rd else "--"
        f.write(f"{D:,} & {v:.3f} & {p} & {q} " + BS * 2 + "\n")
print("note assets written")
