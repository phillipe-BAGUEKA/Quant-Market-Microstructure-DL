# CFM Market Microstructure Challenge

This repository is the foundation for a modular Python, PyTorch, and modern C++
solution to the CFM market microstructure classification challenge. The current
implementation is deliberately limited to project configuration and explicit
schema contracts. It contains no model or training code.

## Data contract

An **event** is one row of the feature table. An **observation** is identified
by `obs_id` and contains an ordered sequence of events. A **class** is the
integer target `eqt_code_cat` associated with one observation.

The following facts were verified locally from the provided CSV files:

- `X_train.csv` contains 16,080,000 events and 160,800 observations.
- Every training observation contains exactly 100 events.
- `y_train.csv` contains 24 classes, numbered from 0 through 23.
- Every class contains 6,700 training observations.
- `X_test.csv` contains 8,160,000 events and 81,600 observations.
- `obs_id` is the key relating feature and target files.
- `eqt_code_cat` is the multiclass target.

The numeric order of `obs_id` is not assumed to represent chronology. Event
order must be preserved from the source until the challenge documentation and
validation protocol establish any stronger rule.

Raw challenge data lives under `data/raw/` and is excluded from Git. The MIT
license in this repository applies to the code; rights and conditions for the
challenge data remain separate and must be checked before redistribution or
transfer to third-party infrastructure.

## Planned architecture

The Python package will progressively add schema validation, chunked readers,
sequence construction, encoding, PyTorch datasets, reproducible training,
evaluation, inference, and submission validation. Planned model families
include simple baselines, ANN, CNN 1D, LSTM, and Transformer architectures;
none is implemented yet.

A future C++ component may handle a genuinely stateful sequential operation if
profiling shows that it is useful. It will require an up-to-date toolchain,
equivalence tests against a Python reference, and a benchmark before any speed
claim. Python integration may later use pybind11 if direct calls are justified.

The project uses scripts and importable modules rather than Jupyter notebooks.
Development and small experiments should run locally first. A CPU or GPU cloud
environment will only be considered after measuring runtime, memory, and data
transfer needs and confirming the applicable challenge rules.

## Local installation

Python 3.12 or newer is required. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

This first project slice has no runtime dependency. Future NumPy, pandas, and
PyTorch dependencies will be declared when package code actually uses them.
For a CPU-only PyTorch environment, the PyTorch CPU index must be recorded
explicitly because a package freeze alone does not preserve the source index.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q
```

The unit tests use only synthetic Python values and temporary paths. They do
not read the raw CSV files.

## Current scope and open points

Implemented now:

- an installable `src/` package;
- an immutable, path-based project configuration;
- constants and validation functions for the observed X and y schemas;
- focused unit tests for configuration and schema validation.

Still to confirm before model selection or final validation are the official
ranking metric, the submission constraints, dependencies between observations,
and any chronology available beyond the row order inside each observation.
Compact data types also require explicit range and precision checks. No score,
trained model, prepared dataset, C++ component, or submission is present.
