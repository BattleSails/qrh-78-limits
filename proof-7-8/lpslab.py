"""Light symbolic test: slab LPs. In a slab S_lo*h <= ell <= S_hi*h every constraint is linear in
v = (b, l_x, ell, s0), with R* >= R_lb(dl, x, S_hi). Minimise s0 per slab; s0 >= 3/4, witnesses dl <= 1/2.
Usage: python lpslab.py S1 S2 ... (slab edges)"""
import sys, numpy as np
from scipy.optimize import linprog
from certgen_rlb import R_lb
z0 = 17/50; K = z0 - 1/6
DL = [n/50 for n in range(1, 26)]; XS = [n/10 for n in range(6)]
def rows(Slo, Shi, k0=1.0, mid=True):
    A = []; names = []
    def add(c0, cb, cl, ce, cs, name): A.append((cb, cl, ce, cs, -c0)); names.append(name)  # cb b+cl lx+ce ell+cs s0 <= -c0
    add(-1, 1, 2, 1, 0, 'struct'); add(0, 0, -1, 1, 0, 'len1'); add(0, 5/6, -1, 1, 0, 'len2')
    add(K, -.5, -K-.5, K, 0, 'small')
    # slab: Slo*h - ell <= 0, ell - Shi*h <= 0, h = 1 - lx + ell
    add(Slo, 0, -Slo, Slo - 1, 0, 'slab_lo'); add(-Shi, 0, Shi, 1 - Shi, 0, 'slab_hi')
    base = (5/6, 1/12, 1/6, -1/6, -1)                       # low without loss, coefficient of s0 = -1
    for (L, de) in (((.25, -.5, -1, .75), 0), ((.25, -.5, -1, 1), -1)):
        bb = (base[0], base[1], base[2], base[3] + de, base[4])
        add(*bb, f'low_d{"0" if de == 0 else "ell"}')
        add(bb[0] + L[0]/2, bb[1] + L[1]/2, bb[2] + L[2]/2, bb[3] + L[3]/2, bb[4], f'low_d{"0" if de == 0 else "ell"}+loss')
    def E(dl, x):
        a = (1 + dl)/2; return (a + K, -a, -K - a, K - .5 + x*dl, -1)
    for dl in DL:
        for x in XS:
            e = E(dl, x)
            co = R_lb(dl, x, Shi, True, k0) + dl/2 - z0
            add(e[0] + co, e[1], e[2] - co, e[3] + co, -1, f'high_h({dl:.2f},{x:.1f})')
            co2 = R_lb(dl, x, 0, False, k0) + dl/2 - z0
            add(e[0] + co2/2, e[1], e[2], e[3], -1, f'high_mid({dl:.2f},{x:.1f})')
        e = E(dl, .5)
        if mid: add(e[0] + (76/75 - 2*dl/3 + dl/2 - z0)/2, e[1], e[2], e[3], -1, f'mid_t1({dl:.2f})')
    e = E(1/50, .5); co = 1 + 1/100 - z0
    add(e[0] + co, e[1], e[2] - co, e[3] + co, -1, 'floor_h'); add(e[0] + co/2, e[1], e[2], e[3], -1, 'floor_half')
    A = np.array(A); return A[:, :4], A[:, 4], names
edges = [float(s) for s in sys.argv[1:]]
for Slo, Shi in zip(edges[:-1], edges[1:]):
    A, bvec, names = rows(Slo, Shi)
    r = linprog([0, 0, 0, 1], A_ub=A, b_ub=bvec, bounds=[(0, None)]*3 + [(0.75, 0.875)], method='highs')
    if r.status == 2: print(f"slab [{Slo},{Shi}]: infeasible for s0 in [3/4, 7/8]"); continue
    duals = -r.ineqlin.marginals; act = [(names[i], round(duals[i], 4)) for i in np.argsort(-duals) if duals[i] > 1e-9]
    print(f"slab [{Slo},{Shi}]: min s0 = {r.fun:.5f} at b,lx,ell = {np.round(r.x[:3],4)}; multipliers {act}")
