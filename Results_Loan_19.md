# Verified results: new 19-input loan model

- Raw dataset rows: 2,260,701
- Eligible resolved loans: 1,345,310
- Test loans: 209,217
- Test period: 2017-02-01 through 2018-12-01
- ROC-AUC: 0.7012
- Average precision: 0.3670
- Charge-off precision: 31.63%
- Charge-off recall: 69.41%
- F1: 0.4345
- Classification threshold (chosen on validation data): 0.167849

At this threshold, about 69% of actual charge-offs were identified. About 32% of flagged loans actually charged off, so false positives remain substantial. The threshold maximizes validation F1; it is not an economically validated lending policy.

| Model | Test ROC-AUC | Test average precision |
|---|---:|---:|
| dummy | 0.5000 | 0.2117 |
| logistic_regression | 0.6878 | 0.3507 |
| lightgbm | 0.7012 | 0.3670 |

The previous notebook's random-split 0.7387 score is not directly comparable: this run uses different predictors, preprocessing, and chronological cohorts.

The API schema has 19 fields. The payment-plan input is disabled as a predictor to avoid possible post-origination leakage, so at most 18 fields are informative. Results are restricted to accepted loans with resolved outcomes in the snapshot. Follow-up differences and selection bias remain; probability calibration and subgroup fairness are not established.

See README.md for startup and integration-test instructions.