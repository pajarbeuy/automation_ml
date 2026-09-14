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

## 7. Final Result & Serving Playground

Show:
- candidate model performance summary
- task-aware validation visualizations (Confusion Matrix for classification, Actual vs Predicted Scatter & Residuals for regression)
- artifact exports (pipeline `.pkl`, training code `train_pipeline.py`, full HTML report)
- **Live Inference Playground (Tab 1)**:
  - dynamic feature form based on candidate model schema
  - live prediction execution
  - classification confidence & class probabilities
  - regression uncertainty cards: $R^2$ confidence score, margin of error ($\pm\text{MAE}$), and expected prediction interval
  - interactive cURL and Python request snippets
- **Batch CSV Prediction (Tab 2)**:
  - upload raw CSV file without target column
  - download enriched CSV with predictions, confidence, and prediction interval bounds

## 8. Model & Data Drift Monitoring Dashboard

Sections:
- dynamic status banner (`HEALTHY`, `WARNING`, `DRIFT_DETECTED`, `INSUFFICIENT_DATA`)
- traffic stat cards (total logged inferences, monitored features, drifted features, drift share %)
- simulation action toolbar (inject normal traffic, inject drifted traffic, refresh, clear logs)
- feature drift breakdown table (method, statistic, p-value/PSI, threshold, mean drift)
- interactive distribution comparison modal (Baseline training vs Current production frequency overlay via Recharts)
