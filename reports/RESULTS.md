# Evaluation results

## Frozen source split and row counts

- Official source training file: 32,561 rows.
- Official source test file: 16,281 rows
  (evaluated after selection).
- The official test outcomes were not used for model fitting or threshold selection.
- The primary global validation threshold was 0.231909.
- Raw archive SHA-256: 7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb.

## Test-set utility

ROC-AUC and average precision are probability metrics. Balanced accuracy and
the other threshold metrics depend on the validation-selected policy. Brier
score and log loss summarize probabilistic prediction error; they do not
establish calibration by themselves.

| model                   | policy                    | roc_auc | average_precision | balanced_accuracy |
| ----------------------- | ------------------------- | ------- | ----------------- | ----------------- |
| primary                 | global                    | 0.904   | 0.758             | 0.820             |
| primary                 | equal_opportunity_by_sex  | 0.904   | 0.758             | 0.810             |
| primary                 | equal_opportunity_by_race | 0.904   | 0.758             | 0.744             |
| no_marital_relationship | global                    | 0.841   | 0.660             | 0.754             |
| no_marital_relationship | equal_opportunity_by_sex  | 0.841   | 0.660             | 0.734             |
| no_marital_relationship | equal_opportunity_by_race | 0.841   | 0.660             | 0.569             |

## Primary model group gaps under the global threshold

The gap is max minus min across groups within each listed audit dimension.
Group-level rates and bootstrap intervals appear in group_metrics.csv;
bootstrap intervals for max-minus-min differences appear in
group_differences.csv.

| group_type | selection_rate_gap | tpr_gap | fpr_gap |
| ---------- | ------------------ | ------- | ------- |
| race       | 0.184              | 0.179   | 0.157   |
| race_sex   | 0.386              | 0.346   | 0.260   |
| sex        | 0.306              | 0.124   | 0.204   |

## Duplicate feature profiles

The source files contain 25 duplicate feature rows
in adult.data and 6 in adult.test. The files
also share 26
unique feature profiles. These official rows are preserved; overlapping
profiles may make the test estimate somewhat optimistic. Counts ignore the
income target.

## Validation-selected policy thresholds

Equal-opportunity-style policies use the lowest validation true-positive rate
under the global threshold as their target. Group-specific thresholds are
selected on validation rows only. They can change error and selection rates
and require group membership at decision time; these diagnostic policies are
not deployment recommendations.

See threshold_policies.csv for every selected threshold and validation
target. The proxy sensitivity excludes marital-status and relationship.

## Limits

This is an educational audit of a historical U.S. benchmark. Its income label
is not a measure of merit or deservingness. Race and sex are excluded from
model features but used for group auditing; correlated proxies remain.
Metrics are unweighted and do not reproduce the Census survey design.
Uncertainty intervals describe resampling variation within this benchmark
split, not source bias or performance in a present-day population. See
docs/METHODOLOGY.md for the full scope and metric definitions.
