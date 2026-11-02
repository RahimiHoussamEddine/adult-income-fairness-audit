# Methodology

## Scope and intended use

The benchmark classifies a historical UCI Adult record as income above or at
most $50,000 per year. This is a statistical audit exercise, not a decision
system. The labels reflect 1994 U.S. Census data and should not be read as
current socioeconomic facts or as an individual measure of merit.

The results are not legal findings and do not establish that a model is fair
or unfair in a real deployment. A valid decision requires a specific use,
affected groups, harm analysis, institutional context, and current data.

## Source and split

- Download the official UCI archive and validate its SHA-256 before use.
- Keep UCI's original adult.data training file and adult.test holdout separate.
- Split adult.data into model-training and policy-validation rows with a
  deterministic, stratified 80/20 split over income, race, and sex.
- Do not inspect adult.test outcomes during fitting or threshold selection.
- Strip the trailing period from labels in adult.test and normalize whitespace.
- Preserve source row counts and audit duplicates and missing values.
- Duplicate checks use feature columns only and do not read the test labels.
- Preserve official rows rather than deduplicating. Shared feature profiles
  across the source training and test files may make holdout performance
  somewhat optimistic; see reports/data_quality.json for the measured count.

The target fields race and sex are excluded from model features. The sampling
field fnlwgt is also excluded from modeling and unweighted metrics; the project
does not model the original complex survey design. Remaining variables can act
as proxies for protected attributes.

## Models

Two regularized logistic regression pipelines are compared:

1. All available non-sensitive features except fnlwgt.
2. A proxy-feature sensitivity that also removes marital-status and
   relationship.

Categorical missing values are imputed with the most common training value and
one-hot encoded. Numeric missing values use the training median and are
standardized. Unknown categories in validation/test are ignored by the encoder.

## Threshold policies

Each model produces probabilities from its training split. A single global
threshold is chosen on validation data to maximize balanced accuracy.

Two additional validation-only policies choose group-specific thresholds:

- Equal-opportunity by sex.
- Equal-opportunity by race.

For each policy, the validation target true-positive rate is the lowest
group-specific true-positive rate achieved by the global threshold. For each
group, choose the highest score threshold that reaches that target on
validation, preferring fewer positive predictions when ties occur. The policy
uses protected-group membership at decision time; it is shown as a diagnostic
trade-off only, not as a deployment recommendation.

Because scores are discrete, a selected group may exceed the common target
slightly. New groups absent from validation do not receive an inferred
threshold; the implementation fails closed and reports the unknown group.

All three policies are evaluated once on the untouched adult.test file. A
validation parity result is not assumed to transfer to the test split.

## Metrics and uncertainty

Probability metrics on the test set include ROC-AUC, average precision, Brier
score, and log loss. Threshold metrics include accuracy, balanced accuracy,
precision, recall, and selection rate. Group metrics include selection rate,
true-positive rate, false-positive rate, precision, accuracy, and support.

Fairlearn MetricFrame reports disaggregated group metrics and max-minus-min
differences. For the primary group-blind model and global threshold, 2,000
nonparametric bootstrap samples provide percentile intervals for group metrics
and between-group differences. These intervals describe sampling variability
within this benchmark split; they do not account for source-data bias, survey
design, label error, or uncertainty about future populations.

Intersectional results are reported with group counts and uncertainty. Small
groups must not be interpreted from a point estimate alone. Confidence
interval overlap is not a test of equality.

## Interpretation limits

- The data are historical, U.S.-specific, and not representative of a current
  labor market or any specific employer.
- The income label is a historical outcome, not a normative target.
- Removing explicit race and sex fields does not remove correlated proxies.
- Different fairness criteria can conflict; no single gap is a universal
  definition of fairness.
- Threshold post-processing can change error rates, selection rates, and the
  sensitive data required at decision time.
- Group-level benchmarks cannot establish individual fairness or legal
  compliance.
- The results are unweighted sample metrics; the project does not model the
  Census survey design represented by fnlwgt.
