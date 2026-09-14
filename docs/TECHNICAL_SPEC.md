# Technical Specification

## 1. Core stack

### Backend
- Python 3.12+
- FastAPI
- Pydantic
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- optional Seaborn

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS

### Persistence
- PostgreSQL

### Experiment tracking
- MLflow in later MVP stages

### Packaging
- Docker
- Docker Compose

## 2. Internal data contracts

### DatasetProfile

```json
{
  "rows": 31500,
  "columns": 18,
  "target": "label",
  "features": [
    {
      "name": "age",
      "dtype": "numeric",
      "missing_rate": 0.021,
      "unique_count": 74,
      "constant": false,
      "skewness": 0.31,
      "outlier_rate": 0.014
    }
  ],
  "duplicates": 17,
  "class_distribution": {
    "negative": 0.71,
    "positive": 0.29
  }
}
```

### Recommendation

```json
{
  "id": "rec_001",
  "category": "missing_value",
  "column": "age",
  "recommended_action": "median_imputation",
  "confidence": 0.86,
  "evidence": [
    "missing_rate=0.021",
    "numeric_feature=true",
    "skewness=0.31",
    "outlier_rate=0.014"
  ],
  "alternatives": [
    "mean_imputation",
    "knn_imputation",
    "drop_rows"
  ],
  "status": "proposed"
}
```

## 3. Pipeline representation

The platform should store a declarative configuration before generating executable sklearn objects.

```json
{
  "numeric": {
    "missing": "median",
    "outlier": "none",
    "scaler": "standard"
  },
  "categorical": {
    "missing": "most_frequent",
    "encoder": "one_hot"
  },
  "model": {
    "name": "logistic_regression",
    "params": {
      "C": 1.0
    }
  }
}
```

### PredictionResult (Serving Contract)

```json
{
  "prediction": 472850.7594,
  "confidence": 0.8273,
  "confidence_metric": "R2 Score",
  "margin_of_error": 27049.62,
  "interval_lower": 445801.14,
  "interval_upper": 499900.38,
  "probabilities": null
}
```

### DriftReport (Monitoring Contract)

```json
{
  "status": "HEALTHY",
  "total_predictions_logged": 120,
  "baseline_rows": 4600,
  "drift_share": 0.0,
  "metrics": {
    "total_features": 12,
    "drifted_features": 0
  },
  "features": [
    {
      "feature_name": "sqft_living",
      "dtype": "numeric",
      "test_name": "kolmogorov_smirnov",
      "statistic": 0.064,
      "p_value": 0.452,
      "threshold": 0.05,
      "wasserstein_distance": 42.15,
      "drift_detected": false
    }
  ]
}
```

## 4. Preprocessing rules

Transformations must be fitted only on training data.

Bad:

```text
Entire dataset
→ impute
→ scale
→ split
```

Correct:

```text
Dataset
→ split
→ fit preprocessing on train
→ transform train
→ transform test
```

For cross-validation:

```text
Fold
├── fit preprocessing on fold-train
├── transform fold-train
├── transform fold-validation
└── evaluate
```

## 5. Model evaluation protocol

### Classification
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
- Primary metrics: `f1_weighted`, `accuracy`, `precision`, `recall`, `roc_auc`
- Baselines: `DummyClassifier`, `LogisticRegression`

### Regression
- Cross-validation: `KFold(n_splits=5, shuffle=True, random_state=seed)`
- Primary metrics: `r2` (coefficient of determination), `neg_root_mean_squared_error`, `neg_mean_absolute_error`
- Baselines: `DummyRegressor`, `LinearRegression`

## 6. Baselines

Every project establishes baseline models to verify that complex algorithms beat trivial predictors:
- Classification: `DummyClassifier(strategy="most_frequent")`
- Regression: `DummyRegressor(strategy="mean")`

## 7. Determinism

Record:
- random seed
- Python version
- dependency versions
- dataset hash
- configuration hash

## 8. Error handling

Every pipeline operation returns structured status:

```json
{
  "status": "failed",
  "stage": "encoding",
  "error_code": "UNSUPPORTED_CARDINALITY",
  "message": "Feature city has 18420 unique categories."
}
```

## 9. Observability

Track:
- job duration
- rows processed
- memory usage where practical
- experiment duration
- model training errors
- recommendation generation errors
