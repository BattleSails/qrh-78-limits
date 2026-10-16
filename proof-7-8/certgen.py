"""Certificate generator for the conservative model (model78c continuous form, kappa_0 = 1).
Float branch-and-bound as in cert78.py, but each leaf records its closing constraint and witness.
Output: JSON with the preorder tree. Verify with certcheck.py (exact rationals, no floats).
Usage: python certgen.py S0_NUM S0_DEN SLO_NUM SLO_DEN OUT.json
"""
import sys, json, time, numpy as np
from fractions import Fraction as Fr
S0 = Fr(int(sys.argv[1]), int(sys.argv[2])); SLO = Fr(int(sys.argv[3]), int(sys.argv[4])); OUT = sys.argv[5]
s0 = float(S0); z0 = 17/50; c = 2/9; MARGIN = 1e-7
NT, NR = 500, 200                       # t step 1/(2 NT) on [1, 3/2]; rho step 1/NR on [0, 1]
DLN = [n for n in range(1, 38) if Fr(n, 50) <= 2*SLO - 1]   # delta = n/50
XN = list(range(6))                                        # x = n/10
SUPN = list(range(0, 61)) + [100]                          # sup = n/100
def R_lb(dl, x, sup, slots):
    q = x*dl
    t = 1 + np.arange(NT + 1)/(2*NT)
    t1, t2 = t[:-1, None], t[1:, None]
    rho = np.arange(NR + 1)[None, :]/NR
    m = (1 - rho)/2; r1, r2 = t1 - 0.5 + rho/2, t2 - 0.5 + rho/2
    ra = np.maximum(r1, 1.0); el = np.where(ra <= 1, 1.0, (1 + 5*ra)/6) - dl*r2
    rb = np.minimum(r2, 1.0)
    zi = np.minimum((1 - r1)/2, sup) if slots else 0.0
    zp = np.minimum(c*(1 - 2*m), sup) if slots else 0.0
    sh = np.minimum(1 - dl*rb - 2*q*zi, 1 - 2*dl*m - 2*q*zp)
    lb = np.where(r1 >= 1, el, np.where(r2 < 1, sh, np.minimum(el, sh)))
    return float(np.min(np.max(lb, axis=1)))
t0 = time.time()
Rns = {(d, x): R_lb(d/50, x/10, 0, False) for d in DLN for x in XN}
Rs = {(d, x, k): R_lb(d/50, x/10, k/100, True) for d in DLN for x in XN for k in SUPN}
print(f"tables {time.time()-t0:.0f}s", flush=True)
def lin_lb(f, B):
    c0, cb, cl, ce = f; (b0, b1), (l0, l1), (e0, e1) = B
    return c0 + (cb*b0 if cb > 0 else cb*b1) + (cl*l0 if cl > 0 else cl*l1) + (ce*e0 if ce > 0 else ce*e1)
K = z0 - 1/6
def E_lin(dl, x):
    a = (1 + dl)/2
    return (a - s0 + K, -a, -K - a, K - 0.5 + x*dl)
def cands(B):
    """All (lower bound, label) pairs for a box."""
    (b0, b1), (l0, l1), (e0, e1) = B
    out = [(lin_lb((-1, 1, 2, 1), B), ['struct']), (lin_lb((0, 0, -1, 1), B), ['len1']),
           (lin_lb((0, 5/6, -1, 1), B), ['len2']),
           (lin_lb((K, -0.5, -K - 0.5, K), B), ['small'])]
    base = (1 - s0 - 1/6, 1/12, 1/6, -1/6)
    out.append((lin_lb(base, B) + max(0, lin_lb((0.25, -0.5, -1, 0.75), B))/2, ['low', 0]))
    out.append((lin_lb((base[0], base[1], base[2], base[3] - 1), B) + max(0, lin_lb((0.25, -0.5, -1, 1), B))/2, ['low', 1]))
    h_lo, h_hi = 1 - l1 + e0, 1 - l0 + e1
    sup_hi = e1/(1 - l1 + e1)
    k = next(n for n in SUPN if n/100 >= sup_hi)
    for d in DLN:
        dl = d/50
        for xn in XN:
            bE = lin_lb(E_lin(dl, xn/10), B)
            co = Rs[(d, xn, k)] + dl/2 - z0
            out.append((bE + (co*h_lo if co > 0 else co*h_hi), ['high_h', d, xn, k]))
            out.append((bE + 0.5*(Rns[(d, xn)] + dl/2 - z0), ['high_mid', d, xn]))
        out.append((lin_lb(E_lin(dl, 0.5), B) + 0.5*(76/75 - 2*dl/3 + dl/2 - z0), ['mid_t1', d]))
    fl = 1/50; a, b_, l_, e_ = E_lin(fl, 0.5); co = 1 + fl/2 - z0
    out.append((lin_lb((a + co, b_, l_ - co, e_ + co), B), ['floor', 'h']))
    out.append((lin_lb(E_lin(fl, 0.5), B) + 0.5*co, ['floor', 'half']))
    return out
tree = []; nleaf = 0
def bb(B, depth):
    global nleaf
    v, lab = max(cands(B), key=lambda p: p[0])
    if v > MARGIN:
        tree.append(['L'] + lab); nleaf += 1; return True
    w = [hi - lo for lo, hi in B]; i = int(np.argmax(w))
    if depth > 80: return False
    tree.append(['S', i])
    lo, hi = B[i]; mid = (lo + hi)/2
    for part in ((lo, mid), (mid, hi)):
        nb = list(B); nb[i] = part
        if not bb(tuple(nb), depth + 1): return False
    return True
ok = bb(((0.0, 1.0), (0.0, 0.5), (0.0, 1.0)), 0)
print(f"s0={S0} S_LO={SLO} ok={ok} leaves={nleaf} {time.time()-t0:.0f}s")
if ok:
    json.dump(dict(s0=[S0.numerator, S0.denominator], s_lo=[SLO.numerator, SLO.denominator],
                   NT=NT, NR=NR, tree=tree), open(OUT, 'w', newline='\n'))
