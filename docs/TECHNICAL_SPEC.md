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

Default:
- stratified K-fold
- 5 folds
- fixed random seed

User chooses primary metric.

Secondary metrics remain visible.

## 6. Baselines

Every project should establish at least:
- DummyClassifier baseline
- Logistic Regression baseline
- one tree-based baseline

This prevents the system from celebrating a model that barely beats random/simple prediction.

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
