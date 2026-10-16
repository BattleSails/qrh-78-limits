\\ Row spectroscopy for u != 1 (PARI side).
\\ For u = x + y w in Z[w], find the primitive Hecke character psi of K = Q(w) with
\\ psi(P) = (u/P)_6 for P prime to 6u (sextic symbol, zeta_6 = 1 + w <-> exp(i pi/3)),
\\ then list every pole of  What(s) D^s / L^S(s, psi),  S = primes dividing 6u,
\\ with 0 <= Re s <= 1/2 and |Im s| <= T, as  [s0, residue coefficient].
\\ Called as  rowspec(x, y, T)  ; writes runs/rowspec_x_y.txt

default(realprecision, 38);
K = bnfinit(w^2 + w + 1, 1);
Wb(t) = my(q = 1 - (2*t - 3)^2); if(q > 1/500, exp(1 - 1/q), 0);
What(s) = intnum(t = 1, 2, Wb(t) * t^(s - 1), 3);

\\ (u/P)_6 as exponent k in 0..5, or -1 if P | u
sextic(u, P) =
{
  my(N = idealnorm(K, P), modP = nfmodprinit(K, P), a, z, b);
  if(idealval(K, u, P) > 0, return(-1));
  a = nfmodpr(K, u, modP); z = nfmodpr(K, 1 + w, modP);
  b = a^((N - 1) / 6);
  for(k = 0, 5, if(z^k == b, return(k)));
  error("no sixth root");
}

findchar(u) =
{
  my(m = idealmul(K, 72, u), R = bnrinit(K, m), cyc = R.cyc, r = #cyc, ps = List(), ks, cands, c);
  forprime(p = 5, 3000, foreach(idealprimedec(K, p), P,
    if(idealval(K, u, P) == 0, listput(ps, [P, sextic(u, P), bnrisprincipal(R, P, 0)]))));
  \\ characters of order | 6: chi = sum c_i e_i / cyc_i with c_i a multiple of cyc_i / gcd(cyc_i, 6)
  my(steps = vector(r, i, cyc[i] / gcd(cyc[i], 6)), lim = vector(r, i, gcd(cyc[i], 6)), found = List());
  forvec(v = vector(r, i, [0, lim[i] - 1]),
    c = vector(r, i, v[i] * steps[i]);
    if(sum(j = 1, #ps, my(t = sum(i = 1, r, c[i] * ps[j][3][i] / cyc[i]) - ps[j][2] / 6); denominator(t) != 1) == 0,
      listput(found, c)));
  if(#found != 1, error("found ", #found, " characters"));
  [R, found[1]];
}

rowspec(x, y, T) =
{
  my(u = x + y*w, RC = findchar(u), R = RC[1], chi = RC[2], cond, Rc, chic, L0, L, zs, out = List(), S, E, Ep, g0);
  my(cr = bnrconductor(R, chi, , 2));
  cond = cr[1]; Rc = cr[2]; chic = cr[3];
  printf("u = %s: cyc %s, chi %s, conductor norm %s, order %s\n", u, R.cyc, chi, idealnorm(K, cond[1]),
         charorder(Rc, chic));
  \\ Euler factors at P | 6u that are unramified for psi: E(s) = prod (1 - psi(P) N(P)^-s)
  S = List();
  foreach(concat([idealprimedec(K, 2), idealprimedec(K, 3), idealfactor(K, u)[, 1]~]), P,
    if(idealval(K, cond[1], P) == 0, listput(S, [idealnorm(K, P), exp(2*Pi*I*chareval(Rc, chic, P))])));
  S = Set(Vec(S));
  printf("  removed unramified Euler factors (N P, psi(P)): %s\n", S);
  E = (s -> prod(j = 1, #S, 1 - S[j][2] * S[j][1]^(-s)));
  Ep = (s -> derivnum(t = s, E(t)));
  if(charorder(Rc, chic) == 1, L0 = lfuncreate(w^2 + w + 1), L0 = lfuncreate([Rc, chic]));
  L = lfuninit(L0, [1/2, 0, T + 2]);
  \\ nontrivial zeros: L not self-dual in general, so zeros of L(s, psi) for both signs of Im
  zs = lfunzeros(L, [-T, T], 64);
  foreach(zs, g, my(rho = 1/2 + I*g); listput(out, [rho, What(rho) / (lfun(L, rho, 1) * E(rho))]));
  \\ s = 0 (trivial zero of L, order 1 for nontrivial psi) and zeros of E on Re s = 0
  if(charorder(Rc, chic) > 1,
    \\ Laurent coefficients at s = 0 by contour integrals (double pole when psi(P) = 1 for a
    \\ removed P); the row "-1 0 c" means c log D.
    g0 = (s -> What(s) / (lfun(L0, s) * E(s)));
    listput(out, [0, intcirc(s = 0, 1/100, g0(s))]);
    listput(out, [-1, intcirc(s = 0, 1/100, s * g0(s))]);
    for(j = 1, #S,
      my(nP = S[j][1], ph = arg(S[j][2]));
      for(k = -12, 12, my(s0 = I * (ph + 2*Pi*k) / log(nP));
        if(abs(s0) > 1e-20, listput(out, [s0, What(s0) / (lfun(L0, s0) * Ep(s0))]))));
  );
  my(fn = Str("runs/rowspec_", x, "_", y, ".txt"));
  write1(fn, "");
  foreach(out, t, write(fn, strprintf("%.25e %.25e %.25e %.25e", real(t[1])*1., imag(t[1])*1., real(t[2])*1., imag(t[2])*1.)));
  printf("  %d zeros with |Im| <= %s written to %s\n", #zs, T, fn);
}
