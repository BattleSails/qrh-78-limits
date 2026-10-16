"""Exact checker for certgen.py certificates. Uses only fractions.Fraction (no floating point).

MODEL (model78c, continuous form, kappa_0 = 1). Unknowns b, l_x, ell >= 0; l_y = l_x + b,
M = l_x + l_y, h = 1 - l_x + ell, z0 = 17/50, kappa = 2 s0 - 1, capacity c = 1/(6 max(3/4, kappa)).
Constraints (all must be <= 0; the paper's strict ones are relaxed, which only enlarges the set):
  struct  M + ell - 1                    len1  ell - l_x          len2  ell - l_x + 5b/6
  small   h(z0 - 1/6) - l_y/2
  low     b/12 + max_{d in [0, ell]} [-d + loss(d)/2] - s0 + 1 - h/6,
          loss(d) = max(0, (1 + 3(ell - d) - 2(M - 2d))/4)
  E(dd; dl, x, R) = a - s0 + h(z0 - 1/6) - a l_y - ell/2 + x dl ell + dd(R + dl/2 - z0), a = (1+dl)/2
  high_h  E(h; dl, x, R*_slots(dl, x, ell/h))   for all dl in [1/50, kappa], x in [0, 1/2]
  high_mid E(1/2; dl, x, R*_none(dl, x))        same range
  mid_t1  E(1/2; dl, 1/2, 76/75 - 2 dl/3)       dl in [1/50, kappa]
  floor   E(h; 1/50, 1/2, 1) and E(1/2; 1/50, 1/2, 1)
  R*(dl, x, sup) = inf_{t in [1, 3/2]} max_{r in [t - 1/2, t]} f(t, r), with m = t - r, q = x dl,
     f = e(r) - dl r  (r >= 1),  e(r) = max(1, (1 + 5r)/6)
     f = min(1 - dl r - 2q zi, 1 - 2 dl m - 2q zp)  (r < 1),
     zi = min((1 - r)/2, sup), zp = min(c(1 - 2m), sup)   [slots]; zi = zp = 0 [no slots].

FACTS USED (each elementary):
 (F1) Domain: struct with b, l_x, ell >= 0 gives b <= 1, l_x <= 1/2, ell <= 1, so the root box covers
      every feasible point.
 (F2) A lower bound of any one constraint that is > 0 on a box excludes the box. Lower bounds of the
      max over d, dl, x are taken at single witnesses (d in {0, ell}; dl = n/50 <= 2 s_lo - 1; x = n/10),
      which is valid for every s0 in [s_lo, s0_cert]: each witness dl <= 2 s_lo - 1 <= kappa is in
      range, every witnessed bound has the form (terms free of s0) - s0, so it only grows as s0
      decreases, and c = 2/9 is constant for s0 <= 7/8.
 (F3) R*(.., sup) is nonincreasing in sup (zi, zp nondecreasing, q >= 0), so R*(ell/h) >= R*(sup_k)
      whenever ell/h <= sup_k on the box.
 (F4) R* >= min_j max_rho L_j(rho): on t in [t_j, t_{j+1}] take r = t - 1/2 + rho/2 (so m = (1-rho)/2),
      r in [r1, r2]; L_j bounds f below using e nondecreasing, dl, q >= 0.
 (F5) In high_h, h * (R + dl/2 - z0) >= h_lo * coef if coef >= 0 else h_hi * coef, coef a lower bound.
Usage: python certcheck.py CERT.json
"""
import sys, json
from fractions import Fraction as Fr
C = json.load(open(sys.argv[1]))
S0 = Fr(*C['s0']); SLO = Fr(*C['s_lo']); NT, NR = C['NT'], C['NR']
assert Fr(1, 2) <= SLO <= S0 <= Fr(7, 8)
z0 = Fr(17, 50); kap = 2*S0 - 1; c = 1/(6*max(Fr(3, 4), kap)); assert c == Fr(2, 9)
K = z0 - Fr(1, 6)
_R = {}
def R_lb(dl, x, sup, slots):
    key = (dl, x, sup, slots)
    if key in _R: return _R[key]
    q = x*dl; best = None
    for j in range(NT):
        t1 = 1 + Fr(j, 2*NT); t2 = 1 + Fr(j + 1, 2*NT); mx = None
        for i in range(NR + 1):
            rho = Fr(i, NR); m = (1 - rho)/2; r1 = t1 - Fr(1, 2) + rho/2; r2 = t2 - Fr(1, 2) + rho/2
            vals = []
            if r2 >= 1:                       # part with r >= 1
                ra = max(r1, Fr(1)); vals.append(max(Fr(1), (1 + 5*ra)/6) - dl*r2)
            if r1 < 1:                        # part with r < 1
                rb = min(r2, Fr(1))
                zi = min((1 - r1)/2, sup) if slots else 0
                zp = min(c*(1 - 2*m), sup) if slots else 0
                vals.append(min(1 - dl*rb - 2*q*zi, 1 - 2*dl*m - 2*q*zp))
            v = min(vals)
            if mx is None or v > mx: mx = v
        if best is None or mx < best: best = mx
    _R[key] = best; return best
def lin_lb(f, B):
    c0, cb, cl, ce = f; (b0, b1), (l0, l1), (e0, e1) = B
    return c0 + (cb*b0 if cb > 0 else cb*b1) + (cl*l0 if cl > 0 else cl*l1) + (ce*e0 if ce > 0 else ce*e1)
def E_lin(dl, x):
    a = (1 + dl)/2
    return (a - S0 + K, -a, -K - a, K - Fr(1, 2) + x*dl)
def bound(lab, B):
    (b0, b1), (l0, l1), (e0, e1) = B
    name = lab[0]
    if name == 'struct': return lin_lb((-1, 1, 2, 1), B)
    if name == 'len1': return lin_lb((0, 0, -1, 1), B)
    if name == 'len2': return lin_lb((0, Fr(5, 6), -1, 1), B)
    if name == 'small': return lin_lb((K, Fr(-1, 2), -K - Fr(1, 2), K), B)
    if name == 'low':
        base = (Fr(5, 6) - S0, Fr(1, 12), Fr(1, 6), Fr(-1, 6))
        if lab[1] == 0:
            return lin_lb(base, B) + max(Fr(0), lin_lb((Fr(1, 4), Fr(-1, 2), -1, Fr(3, 4)), B))/2
        return lin_lb((base[0], base[1], base[2], base[3] - 1), B) + max(Fr(0), lin_lb((Fr(1, 4), Fr(-1, 2), -1, 1), B))/2
    if name in ('high_h', 'high_mid', 'mid_t1'):
        dl = Fr(lab[1], 50); assert Fr(1, 50) <= dl <= 2*SLO - 1      # witness in range for all s0 >= s_lo
    if name == 'high_h':
        x = Fr(lab[2], 10); sup = Fr(lab[3], 100); assert 0 <= x <= Fr(1, 2)
        assert e1 <= sup*(1 - l1 + e1)                                  # ell/h <= sup on the box (F3)
        h_lo, h_hi = 1 - l1 + e0, 1 - l0 + e1; assert h_lo > 0
        co = R_lb(dl, x, sup, True) + dl/2 - z0
        return lin_lb(E_lin(dl, x), B) + (co*h_lo if co >= 0 else co*h_hi)
    if name == 'high_mid':
        x = Fr(lab[2], 10); assert 0 <= x <= Fr(1, 2)
        return lin_lb(E_lin(dl, x), B) + (R_lb(dl, x, Fr(0), False) + dl/2 - z0)/2
    if name == 'mid_t1':
        return lin_lb(E_lin(dl, Fr(1, 2)), B) + (Fr(76, 75) - 2*dl/3 + dl/2 - z0)/2
    if name == 'floor':
        fl = Fr(1, 50); co = 1 + fl/2 - z0; a, b_, l_, e_ = E_lin(fl, Fr(1, 2))
        if lab[1] == 'h': return lin_lb((a + co, b_, l_ - co, e_ + co), B)
        return lin_lb(E_lin(fl, Fr(1, 2)), B) + co/2
    raise ValueError(lab)
tree = C['tree']; pos = 0; leaves = 0; minlb = None; used = {}
def walk(B):
    global pos, leaves, minlb
    node = tree[pos]; pos += 1
    if node[0] == 'S':
        i = node[1]; lo, hi = B[i]; mid = (lo + hi)/2
        for part in ((lo, mid), (mid, hi)):
            nb = list(B); nb[i] = part; walk(tuple(nb))
        return
    v = bound(node[1:], B)
    assert v > 0, (B, node, v)
    leaves += 1; used[node[1]] = used.get(node[1], 0) + 1
    if minlb is None or v < minlb: minlb = v
walk(((Fr(0), Fr(1)), (Fr(0), Fr(1, 2)), (Fr(0), Fr(1))))
assert pos == len(tree)
print(f"VERIFIED: no feasible (b, l_x, ell) for any s0 in [{SLO}, {S0}]; leaves={leaves}, "
      f"smallest closing bound={float(minlb):.3e}, R evaluations={len(_R)}, by constraint={used}")
