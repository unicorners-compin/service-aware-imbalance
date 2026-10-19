# Service-Aware Imbalance: reproducible topology experiments

This code evaluates a declared logistic service-tolerance profile on all ordered
node pairs of a static undirected graph. It reports SAI, reachability, overall
and reachable-pair mean satisfaction, Jain's index, satisfaction Gini, and tails.
The paper is *Service-Aware Imbalance: An Entropy-Based Metric for Functional
Fairness in Complex Networks*.

The experiments are an independent reconstruction with explicit graph settings,
seeds, data provenance, and complete candidate-edge enumeration. See
`docs/theory.md` for the mathematical interpretation and the same-input fairness
comparisons. A low imbalance value must be interpreted with service level and
coverage. Profiles supplied with the examples are model assumptions.

## Environment and quick validation

The full experiments were run on Linux with Python 3.10.12. Install the pinned
packages from PyPI in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use small fixed inputs and need no network access, data service, or
account. They compare the implementation with direct entropy/Jain/Gini formulas,
check disconnected and zero-total cases, batched histograms and edge updates,
calibration, numerical underflow, distance-class identities, unit conversion,
and profile limits.

## Full reconstruction

Run from the repository root. Set thread limits **before** starting Python.
The following executes the full research workload; the test command above is
sufficient for a quick installation check.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/fetch_data.py
python scripts/run_experiments.py --stage synthetic
python scripts/run_experiments.py --stage as --workers 1
python scripts/run_experiments.py --stage benchmark
python scripts/audit_properties.py
python scripts/stress_graphs.py
python scripts/build_report.py
python scripts/summarize_public_results.py
python scripts/plot_stress_cases.py
```

The AS stage processes 41 graphs with 10,670 nodes; the dense benchmark peaked
at about 2.9 GiB process memory on the measured host. Keep at least this much
working memory available with additional operating-system headroom. Runtime
is hardware-dependent. AS workers can be increased if memory permits; benchmark
processes are deliberately sequential. The full workflow uses Linux process
memory/affinity APIs and has not been validated on Windows or macOS.

`configs/full.json` fixes the parameter grids and master seed. Outputs go to
`results/issue-1/`; `issue-1` is a stable directory name and requires no issue
tracker. Raw tables and graph edge lists are under `artifacts/`, and vector and
raster plots under `figures/`. The audit checks 158,753 retained/derived records;
that count is not the number of independent graph samples. A graph reused under
several profiles remains one graph realization.

The representative edge example is chosen using the maximum SAI regret of the
lexicographically selected mean-distance optimum among the declared cases with
disjoint optimal sets. All candidate rows are retained, including agreements
and unfavorable outcomes. It is an exploratory existence example.

## Public input and network access

The Oregon-1 snapshot is downloaded from Stanford SNAP. Its URL, date, and
SHA-256 are pinned in `data/oregon1-provenance.json`. The downloader verifies the
checksum before using the file and reuses a valid local copy. It honors ordinary
`HTTPS_PROXY`, `HTTP_PROXY`, and `NO_PROXY` environment variables when required;
no particular proxy or laboratory service is required by the code. Cite and
follow the dataset provider's terms separately from this source code.

## Published results and figure reproduction

The public repository is https://github.com/unicorners-compin/service-aware-imbalance.
Release `v1.0.0` contains the complete retained result tables: **158,753 records**,
all stored stress-test graph edge lists, distance histograms, seeds and the
benchmark software environment. Both agreeing and disagreeing edge decisions
are included. The original Stanford SNAP download is not redistributed; its
retrieval URL and checksum are in `data/oregon1-provenance.json`.

Download the two assets from the release page, or use the GitHub CLI:

```bash
gh release download v1.0.0 --repo unicorners-compin/service-aware-imbalance --pattern 'sai-retained-results-v1.0.0.tar.gz' --pattern 'SHA256SUMS'
sha256sum -c SHA256SUMS
tar -xzf sai-retained-results-v1.0.0.tar.gz
python scripts/summarize_public_results.py
python scripts/build_report.py
python scripts/build_column_figures.py
```

These commands validate retained data and redraw figures; they do not rerun
simulations. Final column-sized manuscript plots are written to
`results/issue-1/figures/manuscript/`. The mathematical notes use natural
logarithms for raw Theil divergence; the manuscript uses base 2. SAI is invariant
to that log-base choice.

| Paper evidence | Retained files in `results/issue-1/artifacts/` |
|---|---|
| ER, BA, WS, same-input fairness comparisons, structure scatter | `canonical_random.csv` |
| Service-profile and anchor sensitivity | `sensitivity.csv`, `calibration.csv` |
| Weighted costs | `weighted.csv` |
| Dumbbell topology and edge outcomes | `*.edgelist`, `dumbbell_initial.csv`, `edge_candidates.csv`, `edge_selected.csv` |
| Degree-matched controls and modular edge decisions | `stress_metrics.csv`, `stress_decisions.csv`, `stress_edges.csv`, `stress_manifest.json`, `stress_graphs/` |
| Small-graph checks | `exhaustive_small_edge_audit.csv`, `property_audit.json` |
| AS case | `as_results.csv`, `as_histogram_*.json` |
| Runtime and memory | `runtime_memory.csv`, `environment.json` |

The release excludes manuscripts, reviewer correspondence, institutional service
configuration, credentials and third-party literature PDFs. It records the
reported experiments rather than asserting a fresh execution on another host.
No additional reuse license has been specified by the authors for this release.
