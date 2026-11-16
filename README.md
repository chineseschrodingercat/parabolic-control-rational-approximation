# Rational approximation of parabolic control transients

Code and data accompanying *Uniform rational approximation of endpoint transients in linear-quadratic parabolic control*, by **Minhao Liu and Senwei Liang**. The `v1.5.1` tag identifies the manuscript's reproducibility snapshot.

## Contents

| Folder | Contents |
|---|---|
| `experiments/robin` | Robin finite-element model, Riccati and graph checks, time-step/mesh/control-weight experiments, finite-window response fits, and saved outputs |
| `experiments/boundary_zero` | Exact rational 3D model; 80- and 120-digit invariant-graph calculations and exhaustive searches for the first successful complete pole block |
| `certificates/fixed24` | Exact rational 8,804-cell verifier for the Robin subrange certificate and the 24 exact poles |
| `certificates/form_class_counterexample` | Exact rational counterexample to extending that dictionary to the full abstract form class |
| `data/fixed24_search` | Saved finite-mesh search evidence for the Robin-specific conjecture |
| `figures` | Portable plot source, relative input files, PDF, editable SVG and 600-dpi PNG exports |

## Reproduction

Tested with Python 3.13.9; `requirements.txt` records the package versions used for this release. From this repository's root:

```sh
python -m pip install -r requirements.txt
python -B certificates/fixed24/verify_certificate.py
python -B certificates/form_class_counterexample/verify_exact.py
python -B experiments/boundary_zero/compute.py
python -B experiments/robin/run_experiments.py
python -B figures/make_figures.py
```

Each calculation writes outputs beside its script. Figures use the supplied input snapshot; after recomputing the experiments, `python -B figures/sync_inputs.py` updates those inputs before rendering. Floating-point results can vary slightly with the BLAS implementation. The exact-arithmetic verifiers use Python's standard library only. Saved rows include norms, residuals, poles, model parameters and the degree conventions.

## Interpretation

The fixed 24-pole certificate covers `alpha in [0.53,1]`, `beta in [0.1,10]`, `tau=1/320`, `L=kappa=1`, arbitrary observation intervals and conforming Galerkin meshes. The larger Robin control-weight range remains a conjecture supported by finite searches. The counterexample applies to the broader form class.

For the boundary-zero family, the observed degree is the **first** complete block whose two operator norms reach `0.01`. Errors need not decrease from one block to the next. The plotted slope-1/2 line is a normalized scaling guide, while the analytical lower bound is asymptotic with an existential threshold. Computed residuals and precision comparisons are numerical diagnostics; the rational certificates are exact checks.

The repository contains generated mathematical data and the programs needed to reproduce the numerical figures and tables. Article and supplementary-document source files are delivered separately.

## Release history

- `v1.5.1`: corrects the checksum manifest for the exact Git-stored line endings. Scientific code and numerical results are unchanged from `v1.5`.
