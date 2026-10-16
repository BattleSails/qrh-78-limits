"""Exact symbolic lower bounds for the conditional variants MS(kappa_0), audited model (no mid_t1).

Count lemma (x = 1/2, no supply cap, c = 2/9, 2/3 <= kappa_0 <= 1, 0 < dl < 5 kappa_0/6):
  r >= 1:  max_r [e(r) - dl r] = max(1 - dl, 1/6 + t(5 kappa_0/6 - dl) if t > 1/kappa_0),
           e(r) = 1 (r <= 1/kappa_0), (1 + 5 kappa_0 r)/6 (r > 1/kappa_0)
  r < 1:   max_r min(...) = S(t) = 1 - dl(16 + 14t)/37 at r = (28t - 5)/37,  S(3/2) = 1 - dl
  R_inf = S(t*) with t* = (5/6 - 16dl/37)/(5 kappa_0/6 - 23dl/37) if 1/kappa_0 <= t* <= 3/2, else 1 - dl.
Rows (cb, cl, ce, cs, rhs): cb b + cl l_x + ce ell + cs s0 <= rhs; extra rows -v <= 0 for v = b, l_x, ell.
Usage: python symproofk.py   (runs kappa_0 = 1, 97/100, 19/20, 9/10, 4/5 with the LP-chosen supports)
"""
from fractions import Fraction as Fr
z0 = Fr(17, 50); K = z0 - Fr(1, 6)
def R_inf(dl, k0):
    assert Fr(2, 3) <= k0 <= 1 and 0 < dl < 5*k0/6
    t = (Fr(5, 6) - 16*dl/37)/(5*k0/6 - 23*dl/37)
    if 1/k0 <= t <= Fr(3, 2):
        v = 1 - dl*(16 + 14*t)/37
        assert v == Fr(1, 6) + t*(5*k0/6 - dl) and v >= 1 - dl
        r = (28*t - 5)/37; assert t - Fr(1, 2) <= r < 1
        return v
    return 1 - dl
def low(): return (Fr(1, 12), Fr(1, 6), Fr(-1, 6), Fr(-1), Fr(-5, 6))
def struct(): return (Fr(1), Fr(2), Fr(1), Fr(0), Fr(1))
def E_row(dl, x, coef_h):   # E(h; dl, x, R) <= 0 with h(R + dl/2 - z0) = coef_h * h
    a = (1 + dl)/2; kc = K + coef_h
    return (-a, -kc - a, kc - Fr(1, 2) + x*dl, Fr(-1), -(a + kc))
def high_h(dl, k0): return E_row(dl, Fr(1, 2), R_inf(dl, k0) + dl/2 - z0)
def floor_h(): return E_row(Fr(1, 50), Fr(1, 2), 1 + Fr(1, 100) - z0)
NEG = {'b>=0': (Fr(-1), 0, 0, 0, 0), 'lx>=0': (0, Fr(-1), 0, 0, 0), 'ell>=0': (0, 0, Fr(-1), 0, 0)}
def prove(k0, rows):
    names = list(rows); assert len(names) == 4
    M = [[Fr(rows[n][j]) for n in names] + [Fr(0) if j < 3 else Fr(-1)] for j in range(4)]
    for col in range(4):
        piv = next(r for r in range(col, 4) if M[r][col] != 0); M[col], M[piv] = M[piv], M[col]
        M[col] = [v/M[col][col] for v in M[col]]
        for r in range(4):
            if r != col and M[r][col] != 0: M[r] = [a - M[r][col]*b for a, b in zip(M[r], M[col])]
    lam = {n: M[i][4] for i, n in enumerate(names)}
    assert all(v >= 0 for v in lam.values()), (k0, lam)
    comb = [sum(lam[n]*Fr(rows[n][j]) for n in names) for j in range(4)]
    assert comb == [0, 0, 0, -1]
    bound = -sum(lam[n]*Fr(rows[n][4]) for n in names)
    return bound, lam
CASES = {
    Fr(1): {'low': low(), 'struct': struct(), 'H(19/50)': high_h(Fr(19, 50), Fr(1)), 'H(2/5)': high_h(Fr(2, 5), Fr(1))},
    Fr(97, 100): {'low': low(), 'struct': struct(), 'H(19/50)': high_h(Fr(19, 50), Fr(97, 100)), 'H(2/5)': high_h(Fr(2, 5), Fr(97, 100))},
    Fr(19, 20): {'low': low(), 'struct': struct(), 'H(19/50)': high_h(Fr(19, 50), Fr(19, 20)), 'H(9/25)': high_h(Fr(9, 25), Fr(19, 20))},
    Fr(9, 10): {'low': low(), 'struct': struct(), 'H(17/50)': high_h(Fr(17, 50), Fr(9, 10)), 'b>=0': NEG['b>=0']},
    Fr(4, 5): {'low': low(), 'struct': struct(), 'floor_h': floor_h(), 'b>=0': NEG['b>=0']},
}
if __name__ == "__main__":
    for k0, rows in CASES.items():
        bound, lam = prove(k0, rows)
        print(f"kappa_0 = {k0}: s0 >= {bound} = {float(bound):.7f}")
        print("   multipliers:", {n: f"{v} ({float(v):.4f})" for n, v in lam.items()})
    print("Each bound holds for s0 >= 5/6 (all witnesses dl <= 2/5 < 2/3 in range); s0 < 5/6 contradicts low + len1.")

# Struct-free variant: replace (S) M + ell <= 1 by the generalized row-norm cost. With sigma = M + ell - 1,
# paper.tex:8246 gives T_d <= H - 3d + sigma, so the row norm gains (sigma - 3d)_+ and the low estimate
# (L) gains sigma_+/2 at d = 0:  s0 >= 5/6 + b/12 + (l_x - ell)/6 + (b + 2 l_x + ell - 1)/2.
def low_pen():
    return (Fr(1, 12) + Fr(1, 2), Fr(1, 6) + 1, Fr(-1, 6) + Fr(1, 2), Fr(-1), Fr(-5, 6) + Fr(1, 2))
CASES_NOSTRUCT = {
    Fr(1): {'low': low(), 'low+pen': low_pen(), 'H(19/50)': high_h(Fr(19, 50), Fr(1)), 'H(2/5)': high_h(Fr(2, 5), Fr(1))},
    Fr(4, 5): {'low': low(), 'low+pen': low_pen(), 'floor_h': floor_h(), 'b>=0': NEG['b>=0']},
}
if __name__ == "__main__":
    for k0, rows in CASES_NOSTRUCT.items():
        bound, lam = prove(k0, rows)
        print(f"[no struct] kappa_0 = {k0}: s0 >= {bound} = {float(bound):.7f}", {n: str(v) for n, v in lam.items()})
