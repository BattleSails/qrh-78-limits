"""Light infeasibility certificate for the conservative model (model78c, kappa_0 = 1), continuous form.

Certifies: for s0 in [S_LO, S0], no (b, l_x, ell) with b, l_x, ell >= 0 satisfies all constraints
(non-strict versions). Branch-and-bound over (b, l_x, ell); a box is closed when a rigorous lower
bound of one constraint is > TOL on the whole box.

Continuous model: R*(delta, x, sup) = inf_{t in [1, 3/2]} max_{r in [t-1/2, t]} f(t, r), as in
model78.count_t (the code's 801-point r grid and t optimiser approximate this from below).
Lower bound used: R* >= min_j max_rho min_{t in T_j} f(t, r(t, rho)), with r = t - 1/2 + rho/2,
so m = t - r = (1 - rho)/2 is fixed for fixed rho. R* is nonincreasing in sup = ell/h.
Witnesses for the 'for all delta, x' constraints: a fixed grid with delta <= 2 S_LO - 1, so every
witnessed constraint is present for each s0 in [S_LO, S0], and each is decreasing in s0
(the count capacity c = 1/(6 max(3/4, 2 s0 - 1)) is constant 2/9 for s0 <= 7/8).
Floating point with TOL margin; not outward-rounded (light version).
Usage: python cert78.py S0 S_LO
"""
import sys, time, numpy as np
S0, S_LO = float(sys.argv[1]), float(sys.argv[2])
assert S0 <= 7/8
z0 = 17/50; c = 1/(6*0.75); TOL = 1e-9

# ---------- rigorous lower bound for R* ----------
TJ = np.linspace(1, 1.5, 501)               # t-interval endpoints
RHO = np.linspace(0, 1, 201)
def R_lb(dl, x, sup, slots):
    q = x*dl
    t1, t2 = TJ[:-1, None], TJ[1:, None]
    rho = RHO[None, :]
    m = (1 - rho)/2
    r1, r2 = t1 - 0.5 + rho/2, t2 - 0.5 + rho/2  # r ranges over [r1, r2] on the t-interval
    # branch r >= 1: e_long(r) - dl r, e_long nondecreasing
    ra = np.maximum(r1, 1.0)
    el = np.where(ra <= 1, 1.0, (1 + 5*ra)/6) - dl*r2
    # branch r < 1
    rb = np.minimum(r2, 1.0)
    if slots:
        zi_hi = np.minimum((1 - r1)/2, sup)
        zp = np.minimum(c*(1 - 2*m), sup)
    else:
        zi_hi = 0.0; zp = 0.0
    sh = np.minimum(1 - dl*rb - 2*q*zi_hi, 1 - 2*dl*m - 2*q*zp)
    lb = np.where(r1 >= 1, el, np.where(r2 < 1, sh, np.minimum(el, sh)))
    return float(np.min(np.max(lb, axis=1)))

# witnesses
DL = [d for d in np.linspace(0.02, 0.74, 37) if d <= 2*S_LO - 1 + 1e-12]
XS = np.linspace(0, 0.5, 6)
SUPG = np.concatenate([np.linspace(0, 0.6, 61), [1.0, 1e9]])
t0 = time.time()
Rns = {(d, x): R_lb(d, x, 0, False) for d in DL for x in XS}
Rs = {(d, x): np.array([R_lb(d, x, s, True) for s in SUPG]) for d in DL for x in XS}
print(f"R tables {time.time()-t0:.1f}s, {len(DL)} deltas", flush=True)

# ---------- linear lower bounds over a box ----------
def lin_lb(c0, cb, cl, ce, B):
    (b0, b1), (l0, l1), (e0, e1) = B
    return c0 + (cb*b0 if cb > 0 else cb*b1) + (cl*l0 if cl > 0 else cl*l1) + (ce*e0 if ce > 0 else ce*e1)
def E_lin(dl, x, s0):
    """E(dd, ...) without the dd-term, as coefficients (c0, cb, cl, ce)."""
    a = (1 + dl)/2; q = x*dl
    # a - s0 + h(z0-1/6) - a ly - ell/2 + q ell,  h = 1 - lx + ell, ly = lx + b
    k = z0 - 1/6
    return (a - s0 + k, -a, -k - a, k - 0.5 + q)
def addlin(u, v, s=1.0): return tuple(ui + s*vi for ui, vi in zip(u, v))
H = (1.0, 0.0, -1.0, 1.0)                    # h as linear form

def lower_bounds(B, s0):
    (b0, b1), (l0, l1), (e0, e1) = B
    out = {}
    out['struct'] = lin_lb(-1, 1, 2, 1, B)
    out['len1'] = lin_lb(0, 0, -1, 1, B)
    out['len2'] = lin_lb(0, 5/6, -1, 1, B)
    out['small'] = lin_lb(*addlin(tuple((z0 - 1/6)*v for v in H), (0, -0.5, -0.5, 0)), B)
    # low: b/12 + max_d[-d + loss(d)/2] - s0 + 1 - h/6, using d = 0 and d = ell
    base = addlin((1 - s0, 1/12, 0, 0), tuple(-v/6 for v in H))
    # loss(d) = max(0, (1 + 3(ell-d) - 2(M - 2d))/4), M = 2 lx + b
    L0 = (0.25, -0.5, -1.0, 0.75)                 # d = 0
    Le = (0.25, -0.5, -1.0, 1.0)                  # d = ell: (1 - 2M + 4 ell)/4
    low0 = lin_lb(*base, B) + max(0.0, lin_lb(*L0, B))/2
    lowe = lin_lb(*addlin(base, (0, 0, 0, -1)), B) + max(0.0, lin_lb(*Le, B))/2
    out['low'] = max(low0, lowe)
    h_lo, h_hi = 1 - l1 + e0, 1 - l0 + e1
    sup_hi = e1/max(1 - l1 + e1, 1e-12)
    k = np.searchsorted(SUPG, sup_hi)             # first grid sup >= sup_hi
    best_h = best_m = best_f = -9
    for dl in DL:
        for x in XS:
            base_E = lin_lb(*E_lin(dl, x, s0), B)
            # high_h: + h (R + dl/2 - z0) with R >= R_lb(sup_grid[k])
            coef = Rs[(dl, x)][k] + dl/2 - z0
            best_h = max(best_h, base_E + (coef*h_lo if coef > 0 else coef*h_hi)) if base_E > -9 else best_h
            # high_mid: d = 1/2, no slots
            best_m = max(best_m, base_E + 0.5*(Rns[(dl, x)] + dl/2 - z0))
        # mid_t1: d = 1/2, x = 1/2, R = 76/75 - 2 dl/3
        best_f = max(best_f, lin_lb(*E_lin(dl, 0.5, s0), B) + 0.5*(76/75 - 2*dl/3 + dl/2 - z0))
    out['high_h'] = best_h + 0.0
    # high_h above used z0-1/6 inside E_lin and adds h(R + dl/2 - z0): matches E(h, ...)
    out['high_mid'] = best_m; out['mid_t1'] = best_f
    fl = 1/50
    eh = lin_lb(*addlin(E_lin(fl, 0.5, s0), tuple((1 + fl/2 - z0)*v for v in H)), B)
    em = lin_lb(*E_lin(fl, 0.5, s0), B) + 0.5*(1 + fl/2 - z0)
    out['floor'] = max(eh, em)
    return out

# ---------- branch and bound at s0 = S_LO? no: at s0 = S0 (hardest-free end) ----------
def run(s0):
    stack = [((0.0, 1.0), (0.0, 0.5), (0.0, 1.0))]
    closed = 0; used = {}; nmax = 2_000_000
    while stack:
        B = stack.pop()
        lb = lower_bounds(B, s0)
        name = max(lb, key=lb.get)
        if lb[name] > TOL:
            closed += 1; used[name] = used.get(name, 0) + 1; continue
        w = [hi - lo for lo, hi in B]
        i = int(np.argmax(w))
        if w[i] < 1e-7 or closed > nmax:
            return False, B, lb, closed, used
        lo, hi = B[i]; mid = (lo + hi)/2
        for part in ((lo, mid), (mid, hi)):
            nb = list(B); nb[i] = part; stack.append(tuple(nb))
    return True, None, None, closed, used
ok, B, lb, n, used = run(S0)
print(f"s0={S0} witnesses delta<={2*S_LO-1:.3f}: certified={ok} boxes={n} closing={used} {time.time()-t0:.0f}s")
if not ok: print("stuck box", B, {k: round(v, 6) for k, v in lb.items()})
