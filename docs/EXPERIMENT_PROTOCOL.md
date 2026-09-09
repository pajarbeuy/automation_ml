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

Example:

```text
Primary metric: F1
Secondary metrics: precision, recall, ROC-AUC
CV: StratifiedKFold(5)
Seed: 42
```

## 3. Establish baselines

Run:
- DummyClassifier
- Logistic Regression
- tree-based baseline

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
