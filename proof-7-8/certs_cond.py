"""Write every exact certificate of symproofk.py as JSON (cond_certificates.json) and as a LaTeX
table (tab_cond.tex) for the appendix of proof78.tex. Each certificate is re-verified here."""
import json
from fractions import Fraction as Fr
from symproofk import CASES, CASES_NOSTRUCT, prove, R_inf
LABEL = {'low': '(L)', 'struct': '(S)', 'low+pen': r'(L$_\sigma$)', 'floor_h': '(F)', 'b>=0': r'$b\ge0$'}
def lab(n):
    if n.startswith('H('): return r'(H) $\delta=' + n[2:-1] + '$'
    return LABEL[n]
def tex(v): return f'${v.numerator}/{v.denominator}$' if v.denominator != 1 else f'${v.numerator}$'
out, lines = [], []
for tag, cases in (('', CASES), ('struct-free', CASES_NOSTRUCT)):
    for k0, rows in cases.items():
        bound, lam = prove(k0, rows)
        comb = [sum(lam[n]*Fr(rows[n][j]) for n in rows) for j in range(5)]
        assert comb[:4] == [0, 0, 0, -1] and -comb[4] == bound
        rec = {'kappa_0': str(k0), 'variant': tag or 'with (S)', 'bound': str(bound), 'bound_decimal': float(bound),
               'rows': [{'name': n, 'multiplier': str(lam[n]),
                         'coefficients_b_lx_ell_s0': [str(Fr(c)) for c in rows[n][:4]], 'rhs': str(Fr(rows[n][4]))}
                        for n in rows]}
        hs = [n for n in rows if n.startswith('H(')]
        if hs: rec['R_inf'] = {n: str(R_inf(Fr(n[2:-1]), k0)) for n in hs}
        out.append(rec)
        if k0 == 1 and not tag: continue          # already Table 1 in the text
        lines.append(r'\multicolumn{4}{l}{$\kappa_0=' + str(k0) + '$' + (', without (S)' if tag else '') +
                     r': $s_0\ge' + f'{bound.numerator}/{bound.denominator}' + r'$}\\')
        for n in rows:
            rv = tex(R_inf(Fr(n[2:-1]), k0)) if n.startswith('H(') else ''
            lines.append(f'& {lab(n)} & {tex(lam[n])} & {rv}' + r'\\')
        lines.append(r'\addlinespace')
json.dump(out, open('cond_certificates.json', 'w', newline='\n'), indent=1)
open('tab_cond.tex', 'w', newline='\n').write('\n'.join(lines) + '\n')
print(len(out), 'certificates verified and written')
