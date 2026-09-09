# UI Flow

## 1. Dashboard

```text
Projects
[ New Project ]

Project cards
- name
- task
- target
- latest experiment
- status
```

## 2. Dataset setup

```text
Upload CSV
      ↓
Preview
      ↓
Select target
      ↓
Confirm task
```

## 3. EDA workspace

Sections:
- dataset overview
- quality summary
- distributions
- correlations
- target analysis

The UI should distinguish:
- observation
- warning
- recommendation

## 4. Data treatment workspace

Each issue becomes a decision card.

```text
Salary
Missing: 4.2%
Outlier: 7.8%

Recommended:
Median imputation

Why?
...

Alternatives:
[Mean] [KNN] [Drop]

[Accept recommendation]
[Override]
```

## 5. Pipeline workspace

Visual graph:

```text
Raw
 ↓
Imputation
 ↓
Encoding
 ↓
Scaling
 ↓
Feature selection
 ↓
Model
```

Each node should expose its configuration.

## 6. Experiment workspace

Show:
- experiment list
- primary metric
- secondary metrics
- CV mean/std
- training time
- pipeline summary

## 7. Final result

Show:
- selected model
- selected pipeline
- validation metrics
- warnings
- artifact export
- generated report
