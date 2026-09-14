# Testing Strategy

## 1. Unit tests

Test:
- type detection
- missing-value statistics
- skewness calculation
- outlier detection
- recommendation scoring
- pipeline generation
- metric calculation

## 2. Integration tests

Test complete flows:

```text
CSV
→ profile
→ recommendations
→ decisions
→ pipeline
→ experiment
→ artifact
```

## 3. Leakage tests

Create synthetic datasets where leakage is easy to detect.

Verify that:
- imputation is fit per training fold
- scaling is fit per training fold
- encoders are fit per training fold
- feature selection is fit per training fold

## 4. Reproducibility tests

Same:
- dataset
- configuration
- seed
- environment

should produce equivalent results.

## 5. Recommendation tests

For known synthetic scenarios, assert reasonable recommendations.

Example:
- strongly skewed numeric feature + outliers → median should rank above mean

Do not assert that one treatment is universally correct.

## 6. UI tests

Verify:
- upload
- target selection
- recommendation approval
- pipeline preview
- experiment comparison
- artifact export

## 7. Regression tests

Every discovered bug becomes a permanent test where practical.

## 8. Active Test Suite Inventory (79 Passing Tests)

The test suite covers all platform layers without external network or GPU dependencies:
- `test_profiling.py`: Type inference, missing statistics, cardinality, skewness, and constant column handling.
- `test_recommendations.py`: Task-aware recommendation rules (classification vs regression), target skewness, and target outliers.
- `test_pipeline.py`: Leakage-safe ColumnTransformer generation and transformer integrity.
- `test_experiments.py`: Cross-validation execution, metric calculations, and artifact persistence.
- `test_evaluator_regression.py`: Continuous evaluation metrics ($R^2$, MAE, MSE, RMSE) and residual analysis.
- `test_feature_engineering.py`: Automated candidate generation (`AutoFeatureEngineer` log1p skew & interactions).
- `test_tuner.py`: Hyperparameter optimization space and `RandomizedSearchCV` execution.
- `test_export.py`: Standalone Python training script and HTML report generators.
- `test_serving.py`: Live single and batch prediction, task-aware outputs, confidence metrics, and prediction intervals.
- `test_drift_detector.py`: Kolmogorov-Smirnov test, Wasserstein distance, Population Stability Index, and aggregation.
- `test_robustness_serving.py`: Edge cases (empty payload, missing features, unseen categories, invalid types, dropped columns).
- `test_robustness_datasets.py`: Pre-flight dataset validation (empty targets, constant targets, row thresholds, delimiter parsing).
- `test_scaling_preview.py`: Numerical scaling preview and distribution histogram calculation.
- `test_assistant.py`: Rule-based AI explanation generation for all treatment categories.
