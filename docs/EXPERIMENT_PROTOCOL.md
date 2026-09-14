# Experiment Protocol

## Objective

Ensure that model comparisons are scientifically meaningful rather than a leaderboard of accidental preprocessing choices.

## 1. Lock the dataset

Record:
- dataset hash
- target
- excluded columns
- dataset version

## 2. Define objective before training

Classification Example:
```text
Task: Classification
Primary metric: F1-weighted
Secondary metrics: accuracy, precision, recall, ROC-AUC
CV: StratifiedKFold(n_splits=5, shuffle=True)
Seed: 42
```

Regression Example:
```text
Task: Regression
Primary metric: R2 Score (coefficient of determination)
Secondary metrics: RMSE, MAE, MSE
CV: KFold(n_splits=5, shuffle=True)
Seed: 42
```

## 3. Establish baselines

Establish trivial and linear baselines before evaluating complex tree ensembles:
- Classification: `DummyClassifier(strategy="most_frequent")` & `LogisticRegression`
- Regression: `DummyRegressor(strategy="mean")` & `LinearRegression`

## 3.1 Hyperparameter Tuning Protocol

- Candidate models can be tuned using `RandomizedSearchCV` over predefined parameter distributions.
- Tuning runs nested cross-validation across folds without data leakage.
- Tuned models record exact parameters and are registered with a `_tune` suffix.

## 4. Compare pipelines

Each experiment differs by an explicit configuration.

Example:

```text
Experiment A
median + one-hot + standard scaling + logistic regression

Experiment B
median + one-hot + no scaling + random forest
```

## 5. Avoid leakage

All preprocessing must be inside the CV pipeline.

Never calculate:
- global mean
- global scaling parameters
- global encoding vocabulary
- feature selection

using validation/test data.

## 6. Report variance

Show:
- mean CV score
- standard deviation
- fold scores where useful

A tiny metric difference should not be presented as meaningful without considering variance.

## 7. Holdout evaluation

After selecting a candidate using CV:
- evaluate once on untouched test data
- do not repeatedly tune against the test set

## 8. Model selection

The system should rank by the user-defined primary objective, but display:
- primary metric
- secondary metrics
- CV variance
- train/test gap where applicable
- complexity/training time

## 9. Final artifact

Store:
- complete pipeline
- configuration
- metrics
- dataset hash
- environment metadata
- model version
