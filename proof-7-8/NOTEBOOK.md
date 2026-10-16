# Re-optimisation of the OAI 7/8 zero-detection geometry under MS(kappa_0)

Lab notebook of the 7/8 analysis, kept in the order the work was done. It supports the closing
paragraph of `../papers/qrh_family_note.pdf` and the note `../papers/proof78.pdf`. The proof scripts are in this
folder; the models, optimisers and light tests it mentions are in `exploration/`. The run logs named
below are not published: every number from them is in the tables here, and rerunning a script recreates its log.
It is a reduced model of Part II of the OpenAI quasi-RH paper. None of the optima is certified
global. Typical margins are about 1e-6.

## Models
| Model (optimiser) | Content |
|---|---|
| `model78.py` (`opt78.py`, `fine78.py`) | Optimistic model: the intermediate no-slot bound is optimised over t |
| `model78c.py` (`fine78c.py`, `fine78c2.py`) | Conservative model: the intermediate bound only at t = 1, R = 76/75 - 2*delta/3, as stated in the paper |
| `model78h.py` (`fineh.py`, `fineh_fast.py`) | Split by the size of H, with a free ceiling h' on the Poisson frequencies |
| `model78r.py` (`finer.py`) | Common-contour floor, with threshold d1 and a hypothetical count exponent rho |
| `model78d.py` (`fined.py`) | Flawed and abandoned (top-dyad idea, withdrawn) |

## Regression tests (all pass)
1. kappa_0 = 1 reproduces the certificate (Lemma 20.2): s0 = 0.87497.
2. At the paper's geometry, `high_h` matches the paper's slack: -5.5e-3, -1.45e-2 and -2.05e-2 at kappa_0 = 0.97, 0.9 and 0.8.
3. Disabling the new input gives the same result as kappa_0 = 1.

## Results (best boundary s0)
| kappa_0 | Optimistic | Conservative | Free h' | Floor saving, rho = 0.9 or 5/6 (opt / cons) |
|---|---|---|---|---|
| 1.0 | 0.87497 | 0.87497 | 0.87498 (h' = h_max) | 0.87538 / infeasible |
| 0.97 | 0.87386 | 0.87386 | | |
| 0.95 | 0.87312 | 0.8732 | | |
| 0.9 | 0.87117 (b = 0, but the Gram bound needs b > 0) | 0.87224 | | |
| 0.8 | 0.8698 | 0.87224 | 0.86932 | 0.86843 / 0.87079 |

Round conservative point: b = 0.075, l_x = 0.3736, ell = 0.1778, s0 = 0.8726. All constraints
hold there with margins of at least 1.4e-4 on a 200 x 41 grid.

## Conclusions
- In the conservative model the low estimate, the floor bin 51/100 and the t = 1 intermediate
  bound are binding. The high exponent, the only place MS enters, has slack.
- In the rho runs, rho has no effect. The floor has large slack there, and the low estimate
  together with the high exponent binds.
- Lemmas in the paper that hard-code 7/8 and would need re-proving for s0 < 7/8: the second
  Euler region (x_r >= 7/8), the principal residue contour (sigma_0 in [7/8, 1)) and
  epsilon_0 = 3/8.

The run logs (not published) are named by model, kappa_0 and variant.

## Signal sensitivity (`gamma78.py`, logs `gam_<variant>_<gamma>.log`, kappa_0 = 1)
The signal h/6 in C(sigma0) is replaced by gamma*h. Variant `all` changes C everywhere it is
subtracted (consistent with paper.tex:6151); variant `low` changes only the low estimate.
| gamma | 1/6 | 0.18 | 0.20 | 0.22 | 0.25 | 2/7 | 1/3 | 0.4 |
|---|---|---|---|---|---|---|---|---|
| s0 (`all`) | 0.87496 | 0.8640 | 0.8475 | 0.8308 | 0.8059 | 0.7761 | 0.7361 | 0.6801 |
| s0 (`low`) | 0.87503 | 0.8663 | 0.8535 | 0.8415 | 0.8245 | 0.8054 | 0.7817 | 0.7502 |
Break-even gamma (`all` / `low`): 0.87 -> 0.173 / 0.177; 0.85 -> 0.197 / 0.206; 0.83 -> 0.221 / 0.240.
`all` is close to s0 = 7/8 - h*(gamma - 1/6) with h about 0.81. Binding: low, struct, floor and high_h
up to gamma = 0.25; from gamma = 2/7, len2 (l_y >= ell + 11b/6) replaces high_h.
Caveats: below s0 = 7/8 the count lemma's kappa >= 3/4 is violated (the model clamps kappa at 3/4),
and the lemmas that hard-code 7/8 apply. At nd = 80, high_h is up to +1.2e-3 at some optima, so the
s0 values carry about 1e-3 of grid error. The `low` run at gamma = 0.5 crashed (x0 out of bounds).

## Light tests for an order-k probe (coarse grid; `z0light.py`, `modelk.py`, logs `z0l_*.log`, `mk_*.log`)
`modelk.py` replaces every sextic constant by its naive order-k value: signal h/k, Gram b/(2k)
and (2 - 1/k) b (this is the generic cubic large sieve, P_a^{1/3} in place of P_a^{1/6}),
e_long (1+(k-1)r)/k, capacity 1/(k kappa), z0 = (17/50)(6/k). Coarse s0 (k=6 reference 0.8757):
| k = 3, sextic value kept for | none | Gram | z0 | capacity | e_long | all but signal | signal |
|---|---|---|---|---|---|---|---|
| s0 | 0.7721 | 0.7539 | 0.7447 | 0.7721 | 0.7721 | 0.7371 | infeasible < 0.99 |
Costs against the signal-only 0.737: Gram 0.018, z0 0.027; the counts are not binding. All-cubic
optimum b = 0.266, l_x = 0.304, ell = 0.124 (binding low, small, struct). Not derived from a cubic
proof; the count lemma (kappa >= 3/4) and the 7/8-specific lemmas are not valid below 7/8.

## Count-lemma kappa test (`kappatest.py`, logs `kt_*.log`)
All-cubic modelk at fixed s0 with kappa = 2 s0 - 1, clamp at 3/4 versus none: identical margins.
Count-driven margins (high_h, high_mid) at s0 = 0.85 / 0.83 / 0.772: -0.175/-0.236, -0.147/-0.210,
-0.067/-0.131. Binding: low, small, struct. The counts are slack in the model.
Structural check of Lemma `plain` (paper.tex:12940-12985), by hand: the comparison step needs
L + z < (1/2 - 1/k) M with z < M/(k(k kappa - 1)). k = 6, L = M/4: holds for kappa > 1/2.
k = 3 (threshold 2M/3 in place of 5M/6): fails at L = M/4 even for z = 0; needs L < M/6, and with
the slot factor k kappa = 3 kappa the slot bound z < M/(3(3 kappa - 1)) exceeds M/6 for all kappa <= 1.

## Two-branch induction for order k (by hand, paper.tex:12940-12985 and 14750-14772)
Assume the uncentered threshold is theta M with theta = 1 - 1/k (paper.tex:12591: it comes from
counting the sixth-power form). Comparison branch: A_comp <= M - A + 2L + 2z < theta M needs
L + z < (theta - 1/2) M = (1/2 - 1/k) M. Centered branch: deficit A - theta M - s v - (L - v)_+
<= (A - M)_+ at A = M needs L >= M/(k min(s, 1)) >= M/k, whatever the common-support coefficient s
(s = 2/3 for k = 6). Overlap needs 1/k < 1/2 - 1/k, so k > 4. k = 6: [M/4, M/3), the paper's
L = M/4. k = 3: L >= M/3 against L < M/6, disjoint for every s. The cubic transplant of
Lemma `plain` cannot close; it needs a different fourth-moment argument.

Correction (later the same day; `exploration/order_k_ledger.py`, exact scan over A and z): the
k > 4 condition assumes one comparison length L for every product length A, as in the paper.
If L may depend on A, the conditions are L >= (A - theta M)/min(s, 1) and
L < min((A - (1 - theta) M)/2 - z, (A - (k kappa + 1) z)/2). At A = M this needs min(s, 1) > 2/(k - 1):
| k | fixed L | adaptive L |
|---|---|---|
| 3 | fails (any s <= 1) | fails (any s <= 1) |
| 4 | fails | needs s > 2/3 (s = 2/3 is borderline) |
| 5 | needs s about 0.8 or more | closes at s = 2/3 (margin +0.10) |
| 6 | closes | closes |
The cubic conclusion stands. Orders 4 and 5 are blocked instead by the explicit theta coefficients
the reflection identity needs; those are known only for orders 2 and 3, which make up 6 = 2*3.
This is exponent-level only, and the other uses of L in the proof were not checked for an A-dependent choice.

## Direct cubic large sieve on Lemma `plain` (by hand)
Rows k with q_k <= Q = Z^m, normalized product of length N = Z^A, target Z^M, M = m + q.
Generic cubic bound Q + N + (QN)^{2/3}, plus cube copies k = c^3 k' (about Q^{1/3} N).
Valid range: A <= min(M, 3M/2 - m, M - m/3); for q = 0: A <= M/2 from (QN)^{2/3}
and A <= 2M/3 from the copies. Needed: A up to M (zero slots, after reflection) and A + (k kappa - 1) z <= M.
Deficit at A = M: (m - q)/3 from (QN)^{2/3} and m/3 from the copies, so m/3 (M/3 when q = 0). Proportional to the width, so not absorbable.

## Light certificate for the conservative model, kappa_0 = 1 (`cert78.py`, logs `cert_*.log`)
Branch-and-bound over (b, l_x, ell) >= 0 with exact lower bounds of each constraint on a box
(affine pieces exactly; R* = inf_t max_r of the continuous count, bounded below on t-intervals;
R* nonincreasing in ell/h). The 'for all delta, x' constraints use witnesses delta <= 2 S_LO - 1,
so one run covers s0 in [S_LO, S0] (all witnessed constraints decrease in s0; capacity constant 2/9).
Strict inequalities are relaxed to non-strict. Floating point with margin 1e-9, not outward-rounded.
| Run | Covers | Result | Boxes |
|---|---|---|---|
| 0.75, S_LO = 0.5 (no witnesses) | (1/2, 0.75] | infeasible | 10 |
| 0.874, S_LO = 0.75 | [0.75, 0.874] | infeasible | 1013 |
| 0.8749, S_LO = 0.75 | [0.75, 0.8749] | infeasible | 27910 |
| 0.87495 | — | not certified, near-feasible box b = 0.1456, l_x = 0.3438, ell = 0.1669 | |
| 0.875 (sanity) | — | not certified, near-feasible box | |
Conclusion: within this model no geometry gives s0 <= 0.8749; the model's limit is in (0.8749, 0.87495].
Closing constraints: struct, low, high_h, floor (len1, len2 for edge boxes).

## Exact certificate (`certgen.py` -> JSON, `certcheck.py` verifies with Fractions only)
`certcheck.py` states the model and the five elementary facts used (domain, witnesses, monotonicity
of R* in ell/h, the t-interval lower bound for R*, the sign rule for h * coef). It rebuilds the
dyadic subdivision tree, checks that it is complete, and recomputes every closing bound exactly.
- `cert_750.json`: VERIFIED for s0 in [1/2, 3/4], 10 leaves, smallest bound 2.08e-2.
- `cert_874.json`: VERIFIED for s0 in [3/4, 437/500], 1013 leaves, smallest bound 6.46e-6,
  71 exact R* bounds, about 80 s.
Negative controls (relabeled leaf; s0 = 7/8 with the 3/4 tree) are rejected.
So in this model no geometry reaches s0 <= 0.874. The 0.8749 / 0.87495 bracket is still float-only.

## Symbolic proof (`symproof.py`, exact Fractions)
Lemma: for x = 1/2 and no cap, R*(dl) = 1 - dl(16 + 14t*)/37, t* = (5/6 - 16dl/37)/(5/6 - 23dl/37)
(r >= 1 branch increasing in r; r < 1 branch maximal at r = (28t - 5)/37). Caps never bind above 1/4
(z_i <= 1/4, z_p <= 2/9), and R* is nonincreasing in the cap, so R*(ell/h) >= R*_inf. Matches
model78.Rstar to 6e-5 (grid error). In the model's variables:
  (L)  s0 >= 5/6 + b/12 + (l_x - ell)/6                      [low, d = 0, loss dropped]
  (S)  2 l_x + b + ell <= 1                                   [struct]
  (H1) s0 >= (172483/124275)(1 - l_x) + (48208/124275) ell - (69/100) b   [high_h, dl = 19/50, x = 1/2]
  (H2) s0 >= (13526/9735)(1 - l_x) + (3791/9735) ell - (7/10) b            [high_h, dl = 2/5, x = 1/2]
s0 < 5/6 contradicts (L) with len1 (l_x >= ell) and b >= 0. For s0 >= 5/6 both witnesses are in range,
and the combination with multipliers 4307228/5301659, 331477/5301659, 1706710/15904977, 116053/1445907
(L, S, H1, H2) gives s0 >= 4638705/5301659 = 0.8749535...
Result: every feasible geometry of the model has s0 >= 0.8749535; this replaces the 1013-box certificate.

## Source-to-model audit (independent reader, 2026-10-16)
low MATCHES (8564-8642, 8306-8310); struct WEAKER (paper fixes M + ell = 1); high_h MATCHES/WEAKER
(d = h is worst on [1/2, h], line 16133; R* = inf_t max_r is the right direction, smaller R helps the
proof); clamp kappa >= 3/4 exactly right (lem:plain); len1 WEAKER. Stronger than the paper needs:
mid_t1 (76/75 constant, +1/150) and len2 (only makes P_a^{1/6} dominate). Neither is used by
`symproof.py`, and z0 cancels in the high_h rows at d = h, so the symbolic bound s0 >= 0.8749535 uses
only constraints the audit marks MATCHES or WEAKER. The 1013-box certificate used len2 in 31 boxes.

## Conditional variants, exact (`symproofk.py`) and write-up (`proof78.tex/.pdf`, 6 pp)
Count lemma with MS(kappa_0): R_inf = 1 - dl(16 + 14t*)/37, t* = (5/6 - 16dl/37)/(5 kappa_0/6 - 23dl/37)
when 1/kappa_0 <= t* <= 3/2, else 1 - dl. Audited model (no mid_t1). Exact lower bounds:
| kappa_0 | rows | bound | numerical optimum |
|---|---|---|---|
| 1 | L, S, H(19/50), H(2/5) | 4638705/5301659 = 0.8749535 | 0.87497 |
| 0.97 | L, S, H(19/50), H(2/5) | 0.8738687 | 0.87386 |
| 0.95 | L, S, H(19/50), H(9/25) | 0.8731136 | 0.87312 |
| 0.9 | L, S, H(17/50), b >= 0 | 0.8711593 | 0.87117 |
| 0.8 | L, S, floor, b >= 0 | 167/192 = 0.8697917 | 0.8698 |
L, S, floor and b >= 0 involve no count, so s0 >= 167/192 = 7/8 - 1/192 holds for every kappa_0.
With mid_t1 (76/75) the LP bounds rise for kappa_0 <= 0.9 (0.87115, 0.87042), which explains the old
conservative 0.87224 plateau.

## Floor bin / Euler region light test
Floor bound (L, S, F, b >= 0): s0 >= 2/3 + (1 + 2 d0)/(5 + 6 d0), exact; 167/192 at d0 = 1/50, limit 13/15.
Local Euler factors (paper.tex:4144-4170) stay convergent for any x_r = a > 1/2 with margins ~ (x_r - 1/2):
at x_r = 0.505, z = 17/50 the good-prime exponent is -1.07; ramified terms <= 1 - 2x - eps0.
LP lower bounds (audited model, witnesses dl = n/100 <= 0.66, d0 = floor bin):
| d0 | kappa_0 = 1 | 0.9 | 0.8 | 0.7 |
|---|---|---|---|---|
| 0.02 | 0.87491 | 0.87111 | 0.86979 (floor) | 0.86979 (floor) |
| 0.01 | 0.87491 | 0.87111 | 0.86825 (floor) | 0.86825 (floor) |
| 0.004 | 0.87491 | 0.87111 | 0.86813 (high_h(0.17)) | 0.8673 (floor) |
A smaller floor bin gives nothing unconditionally or at kappa_0 >= 0.9; it helps only with MS(kappa_0 <= 0.8),
by at most 1/320. Low alone permits 5/6 (l_x = ell, b = 0); the trivial count R = 1 on the floor bin pins 13/15.

## Is M + ell <= 1 necessary? (struct)
Without struct the LP degenerates (l_x = ell = 1, s0 = 5/6), so struct carries the bounds. The paper uses
M + ell = 1 once (paper.tex:8246, "Since M'+ell'-1=-3d"). With sigma = M + ell - 1 free, T_d <= H - 3d + sigma,
so retained dual dyads give max(H, v + ell_b) <= H + (sigma - 3d)_+; the loss formula does not use sigma.
Hence struct is sufficient, not necessary; exceeding it costs (sigma - 3d)_+ in the row norm, sigma_+/2 in low at d = 0.
With that penalty in place of struct the exact bounds are unchanged (symproofk.py, CASES_NOSTRUCT):
0.8749535 at kappa_0 = 1 and 167/192 at 4/5; optima stay at sigma = 0. M + ell > 1 starts to pay only if
the penalty slope drops below about 0.077 (kappa_0 = 1) or 0.08 (0.8), against the derived 1/2.
Scope: the penalty is the cost of the paper's row-norm argument extended as written; other implicit uses of
M + ell = 1 were not found by the audit.

## Hypothetical bilinear saving in the low separation (LP, audited system)
low -> l_x/2 + b/12 - eta*l_x + ... . LP lower bounds:
| eta | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|
| kappa_0 = 1 | 0.87491 | 0.87344 | 0.87193 | 0.86872 | 0.85830 | 0.84031 | 0.80215 |
| 0.9 | 0.87111 | 0.86946 | 0.86780 | 0.86446 | 0.85429 | 0.83680 | 0.79961 |
| 0.8 | 0.86979 | 0.86815 | 0.86651 | 0.86320 | 0.85312 | 0.83577 | 0.79887 |
eta needed (kappa_0 = 1 / 0.9 / 0.8): 0.87 -> 0.0161 / 0.0034 / 0; 0.85 -> 0.0733 / 0.0625 / 0.0591;
0.83 -> 0.1278 / 0.1189 / 0.1162. At the optima l_x ~ 0.41-0.43, so 0.85 needs an absolute saving of
about Z^{-0.031} on the Cauchy-Schwarz bound. Binding set stays low + struct + one high_h (or floor);
no new constraint appears up to eta = 0.2. Slope ds0/deta ~ -0.33 to -0.36.

## Status: FROZEN (2026-10-16)
Parameter optimisation of the 7/8 architecture is closed. Verified artefacts and their SHA-256 sums are in
`../MANIFEST.sha256` (run `sha256sum -c MANIFEST.sha256` from the repository root). Write-up:
`../papers/proof78.pdf`; summary paragraph in `../papers/qrh_family_note.pdf`. Results constrain the audited architecture only, not other methods.
