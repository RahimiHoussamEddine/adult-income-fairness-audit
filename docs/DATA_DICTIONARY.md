# Data dictionary

## UCI source columns

The Adult source has 14 input attributes and one income target.

| Column | Project handling |
|---|---|
| age | Numeric model feature |
| workclass | Categorical model feature; '?' is missing |
| fnlwgt | Census sampling field; excluded from features and unweighted metrics |
| education | Categorical model feature |
| education-num | Numeric model feature |
| marital-status | Model feature; removed in proxy sensitivity |
| occupation | Categorical model feature; '?' is missing |
| relationship | Model feature; removed in proxy sensitivity |
| race | Sensitive audit attribute; excluded from model features |
| sex | Sensitive audit attribute; excluded from model features |
| capital-gain | Numeric model feature |
| capital-loss | Numeric model feature |
| hours-per-week | Numeric model feature |
| native-country | Categorical model feature; '?' is missing |
| income | Binary target: >50K is 1; <=50K is 0 |

## Checked-in reports

- reports/model_metrics.csv has one row per model and threshold policy on the
  locked test split.
- reports/group_metrics.csv has aggregate per-group rates, support counts, and
  bootstrap interval columns where calculated.
- reports/threshold_policies.csv contains validation-selected thresholds and
  validation TPR targets by sensitive group.
- reports/data_quality.json has source counts, feature missingness, duplicate
  feature profiles, and split evidence. Test label counts are not computed
  before final evaluation.
- reports/summary.json contains headline aggregate metrics and source
  provenance.
- reports/RESULTS.md presents the evaluated trade-offs and limitations.

## Local generated data

The downloaded source archive remains under ignored data/raw. The pipeline
does not persist fitted models, person-level predictions, or source rows.
Public reports contain aggregate metrics only.
