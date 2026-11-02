# Adult Income Fairness & Threshold Policy Audit

An educational audit of how a group-blind income classifier performs across
sex, race, and intersectional groups in UCI's historical Adult Census Income
benchmark. It compares a single global threshold with validation-only
equal-opportunity threshold policies and reports subgroup uncertainty.

The target is the dataset's historical label for annual income above $50,000.
This project does not estimate merit, employability, or deservingness. It is
not suitable for hiring, employment, lending, benefits, or individual decisions.

## Measured result

Run make build to reproduce the results for the pinned UCI source snapshot.
The report separates test-set predictive utility from subgroup selection and
error-rate disparities, with bootstrap intervals for the primary group audit.
The policy comparison uses a locked official test split; group thresholds are
selected on a validation split only.

## Pipeline

~~~mermaid
flowchart LR
    A[UCI Adult source archive] --> B[Schema and missingness audit]
    B --> C[Train and validation split]
    C --> D[Group-blind logistic models]
    C --> E[Validation-only threshold policies]
    D --> F[Locked test evaluation]
    E --> F
    F --> G[Aggregate group metrics and uncertainty]
    G --> H[Dashboard and read-only API]
~~~

## Run locally

Use Python 3.12:

~~~bash
python -m venv .venv
. .venv/bin/activate
make install-dev
make fetch
make build
make verify
make test
make quality
make check-deps
make app
~~~

The raw UCI archive stays in an ignored data path. The pipeline writes no
person-level predictions or fitted models. Only aggregate evaluation metrics,
source provenance, and documentation are checked in. Regenerate the exact
installed package lock with make lock after installing the development
environment.

The aggregate API is read-only:

~~~text
GET /health
GET /summary
GET /groups?model=primary&policy=global&group_type=race
~~~

Run it locally with make api. The dashboard runs with make app.

## What it demonstrates

- Group-blind modeling with protected fields kept for audit and not used as
  prediction features.
- A fixed source test split and validation-only policy selection.
- Disaggregated selection, true-positive, and false-positive rates, including
  intersectional groups and confidence intervals.
- An explicit proxy-feature sensitivity and policy trade-off analysis.
- Data provenance, licensing, tests, packaging, CI, Docker, a dashboard, and a
  read-only aggregate API.

See reports/RESULTS.md, docs/METHODOLOGY.md, and docs/DATA_DICTIONARY.md.
Source attribution and limitations are in NOTICE.md.
