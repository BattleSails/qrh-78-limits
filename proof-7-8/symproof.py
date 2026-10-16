"""Short symbolic infeasibility proof for the conservative model (kappa_0 = 1), exact rationals.

Uses four inequalities of the model (see certcheck.py docstring for the model):
  (L)  low at d = 0:      b/12 - s0 + 1 - h/6 + max(0, (1 + 3 ell - 2M)/4)/2 <= 0   (we drop loss(0) >= 0)
  (S)  struct:            2 l_x + b + ell - 1 <= 0
  (H1),(H2) high_h at x = 1/2 and dl = 19/50, 20/50:
       a - s0 + h(z0 - 1/6) - a l_y - ell/2 + (dl/2) ell + h (R + dl/2 - z0) <= 0,  a = (1+dl)/2,
       with R = R*(dl, 1/2, ell/h) >= R_inf(dl) (R* nonincreasing in the cap; caps never bind above 1/4).
Closed form (x = 1/2, no cap, c = 2/9):
  r >= 1 branch: max_r = 1/6 + t (5/6 - dl);  r < 1 branch: max_r min(...) = 1 - dl (16 + 14 t)/37
  at r = (28 t - 5)/37 in [t - 1/2, 1); R_inf = value where they meet, t* = (5/6 - 16dl/37)/(5/6 - 23dl/37).
Farkas: lambda >= 0 with sum lambda_i * (row_i) having zero b, l_x, ell coefficients and s0
coefficient -1 gives s0 >= (bound). Valid for s0 >= (1 + 20/50)/2 = 7/10 (witnesses in range).
"""
from fractions import Fraction as Fr
z0 = Fr(17, 50); K = z0 - Fr(1, 6); c = Fr(2, 9)
def R_inf(dl):
    t = (Fr(5, 6) - 16*dl/37)/(Fr(5, 6) - 23*dl/37)
    assert 1 <= t <= Fr(3, 2)
    v1 = Fr(1, 6) + t*(Fr(5, 6) - dl); v2 = 1 - dl*(16 + 14*t)/37
    assert v1 == v2
    r = (28*t - 5)/37; assert t - Fr(1, 2) <= r < 1
    return v1, t
# rows as (cb, cl, ce, cs, rhs) meaning cb b + cl l_x + ce ell + cs s0 <= rhs ; h = 1 - l_x + ell
def row_low():               # b/12 - s0 + 1 - h/6 <= 0  (low at d = 0, dropping loss(0) >= 0)
    return (Fr(1, 12), Fr(1, 6), Fr(-1, 6), Fr(-1), -(1 - Fr(1, 6)))
def row_struct():
    return (Fr(1), Fr(2), Fr(1), Fr(0), Fr(1))
def row_high(dl):
    a = (1 + dl)/2; R, _ = R_inf(dl); co = R + dl/2 - z0
    # a - s0 + K h - a(l_x + b) - ell/2 + (dl/2) ell + co h
    kc = K + co
    return (-a, -kc - a, kc - Fr(1, 2) + dl/2, Fr(-1), -(a + kc))
rows = {'L': row_low(), 'S': row_struct(), 'H1': row_high(Fr(19, 50)), 'H2': row_high(Fr(20, 50))}
names = list(rows)
# solve sum lam_i * (cb, cl, ce, cs) = (0, 0, 0, -1) exactly (Gauss elimination over Fractions)
M = [[rows[n][j] for n in names] + [Fr(0) if j < 3 else Fr(-1)] for j in range(4)]
for col in range(4):
    piv = next(r for r in range(col, 4) if M[r][col] != 0); M[col], M[piv] = M[piv], M[col]
    M[col] = [v/M[col][col] for v in M[col]]
    for r in range(4):
        if r != col and M[r][col] != 0: M[r] = [a - M[r][col]*b for a, b in zip(M[r], M[col])]
lam = {n: M[i][4] for i, n in enumerate(names)}
assert all(v >= 0 for v in lam.values()), lam
bound = -sum(lam[n]*rows[n][4] for n in names)        # -s0 <= sum lam rhs  =>  s0 >= -sum lam rhs
for dl in (Fr(19, 50), Fr(20, 50)):
    R, t = R_inf(dl); print(f"R_inf({dl}) = {R} ~ {float(R):.6f} at t* = {t}")
print("multipliers:", {n: str(v) for n, v in lam.items()}, {n: round(float(v), 5) for n, v in lam.items()})
print(f"PROVED: every feasible geometry with s0 >= 7/10 has s0 >= {bound} ~ {float(bound):.7f}")
