# Limits of the 7/8 zero-free-region architecture

This repository contains two research notes and their code. They study the sextic family behind
OpenAI's quasi-Riemann Hypothesis preprints, and how far the exponent architecture of the
7/8 zero-free-region proof can be pushed.

The main result is a short, exactly verified lower bound: within the audited inequalities of
that proof, no choice of geometry gets below **0.8749535**, so the published 7/8 is within
5 · 10⁻⁵ of its own limit.

> **Scope.** These results limit one proof architecture. They aren't barriers for other
> methods, and nothing here proves or disproves the Riemann Hypothesis.

## Main results

| Result | Statement | Status |
|---|---|---|
| Unconditional limit | s₀ ≥ 4638705/5301659 ≈ 0.8749535 for every admissible geometry | Exact proof from four audited inequalities |
| Conditional limits | Under MS(κ) for all κ ∈ [κ₀, 1]: 0.87387, 0.87311, 0.87116 for κ₀ = 0.97, 0.95, 0.9 | Exact proofs |
| Mean-square floor | s₀ ≥ 167/192 = 7/8 − 1/192 for every κ₀ | Exact proof; independent of the count |
| Robustness | Dropping M + ℓ ≤ 1 leaves both bounds unchanged; a smaller floor bin gains at most 1/320 | Exact proofs |
| Cubic probe | The fourth-moment induction closes only for character order k > 4 | Exponent-level argument |
| Sextic family numerics | The mean square matches square-root cancellation for row ranges H < D, up to D = 320,000 | Numerical |

## Status of the underlying results

- **The 7/8 theorem.** OpenAI's repository reports that the 7/8 zero-free half-plane is
  formalized in Lean, for ζ, every Dirichlet L-function and the finite-order Hecke L-functions
  over ℚ(√−3) (see `lean/docs/003.md` in [github.com/openai/math](https://github.com/openai/math)).
  This repository hasn't built or checked that Lean code.
- **What the limit depends on.** The lower bound s₀ ≥ 0.8749535 is a statement about the
  inequalities the paper's argument uses, so it holds whether or not that proof is correct. Its
  significance comes from the 7/8 theorem: it says the method behind that theorem is within
  5 · 10⁻⁵ of its own limit.
- **What was audited.** The inequalities were checked against the paper's LaTeX source (pinned
  below), not against the Lean formalization. If the formal proof follows a different route or
  uses different constants, these results describe the paper's argument.
- **The conditional results** need the mean-square hypothesis MS(κ) for κ < 1, which nobody has
  proved. They become usable only if it is proved, and even then 167/192 is the most this
  method can give.
- **The 11/12 companion,** whose mean-square estimate the family note uses, isn't listed in that
  formalization document.

## Papers

| File | Content |
|---|---|
| [`papers/proof78.pdf`](papers/proof78.pdf) | *A four-inequality limit for the Part II geometry of the 7/8 zero-free region* (6 pages). The bounds above, with every multiplier listed exactly. |
| [`papers/qrh_family_note.pdf`](papers/qrh_family_note.pdf) | *A numerical look at the sextic family in the quasi-Riemann argument* (13 pages). The family numerics, the conditional exponent ½ + 5κ/12, and a summary of the 7/8 analysis. |

LaTeX sources are in `papers/`.

## Repository layout

```text
papers/                  The two notes (PDF and LaTeX)
proof-7-8/               Exact proofs and certificates for the 7/8 architecture
  symproofk.py           Main proof: all lower bounds in exact rational arithmetic
  certs_cond.py          Re-verifies every certificate and writes cond_certificates.json
  certgen.py, certcheck.py, cert_*.json
                         Independent box certificate and exact checker (kappa_0 = 1)
  lpslab.py, certgen_rlb.py
                         LP relaxations used to find the certificates
  NOTEBOOK.md            Lab notebook with every table, in the order the work was done
  exploration/           Reduced models, optimisers and light tests (numerical, not certified)
family-numerics/         Arithmetic of Z[ω], the sextic family, zero statistics, spectroscopy
MANIFEST.sha256          Checksums of the papers and the proof artefacts
```

## Reproduce the main results

Each check takes from under a second to about two minutes.

```bash
pip install -r requirements.txt
cd proof-7-8
python symproofk.py               # proves all 8 lower bounds exactly
python certs_cond.py              # re-verifies the 7 certificates, writes the JSON
python certcheck.py cert_750.json # box certificate, s0 in [1/2, 3/4]
python certcheck.py cert_874.json # box certificate, s0 in [3/4, 437/500], about 80 s
```

To check that the files are unchanged, run `sha256sum -c MANIFEST.sha256` from the repository root.

## Source pinning

Line numbers in the notes and code refer to `paper.tex` of the preprint *The Quasi-Riemann
Hypothesis: A Zero-Free Half-Plane Re s > 7/8* (September 30, 2026), in
[github.com/openai/math](https://github.com/openai/math), under
`preprints/The-Quasi-Riemann-Hypothesis-September-30-2026/build/paper.tex`. The version used has

```text
SHA-256  42a5ee0febca59fd1def55cfd6c6808c322ef4237d7726303ea52f711deac6a3
```

That file isn't included here. Download it from the OpenAI repository.

## Requirements

- Python 3.13 with NumPy, SciPy, Numba, mpmath and Matplotlib (see `requirements.txt`).
  The proof scripts need only NumPy and SciPy; the exact checks use the standard `fractions` module.
- PARI/GP 2.19, only for the row L-function spectroscopy in `family-numerics/`. If `gp` isn't
  on your `PATH`, set the `GP` environment variable to the executable.
- A LaTeX distribution, only to rebuild the notes.

## Data

The family computations write their arrays to `family-numerics/runs/` (about 370 MB in total,
with files up to 42 MB). They aren't in the repository. To regenerate them, run the scripts. For
example, `python family.py 80000 1.1 runs/D80000.npz` computes one column scale. Each script
documents its usage in its docstring.

## How the results were checked

- An independent reader derived each inequality used in the proofs from the OpenAI source,
  without seeing the code. Each one matches the paper or is weaker than it.
- Two methods agree on the κ₀ = 1 bound: the four-inequality proof and a 1,013-box certificate
  checked in exact arithmetic.
- The checker rejects tampered certificates.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the notes, not individual scripts.

## License

The code is under the MIT License ([`LICENSE`](LICENSE)). The notes in `papers/` are under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([`papers/LICENSE`](papers/LICENSE)).

## Acknowledgments

Computations and drafting were assisted by Claude (Anthropic), an AI system. All results were
checked by direct computation.
