# AutoML Human-in-the-Loop

An evidence-driven machine learning workflow platform that automates repetitive tabular ML work while keeping consequential decisions under human control.

## Core idea

The system does not try to replace the data scientist. It:

1. Profiles a dataset automatically.
2. Generates EDA visualizations and statistical findings.
3. Detects potential data-quality issues.
4. Recommends treatment options with evidence and confidence.
5. Lets the user approve or override those recommendations.
6. Builds reproducible preprocessing pipelines.
7. Runs controlled model experiments.
8. Compares pipelines using configurable objectives.
9. Tracks experiments and decisions.
10. Produces a reproducible final pipeline and model artifact.

## Platform Scope

The platform targets **tabular machine learning** (Classification and Regression) with CSV input, structured across Phases 0 to 8 of the product roadmap.

Supported features:
- Evidence-driven dataset profiling and statistical detection
- Task-aware continuous and categorical EDA visualization
- Preprocessing recommendations (missing values, scaling, encoding, target transformations)
- Explicit human approval, rejection, and override of treatments
- Leakage-safe Scikit-Learn Pipeline and ColumnTransformer construction
- Multi-model benchmarking (linear, tree-based, gradient boosting, KNN, XGBoost, LightGBM)
- Task-aware cross-validation (StratifiedKFold for classification, KFold for regression)
- Automated feature engineering (`AutoFeatureEngineer` log1p skew & numeric interactions)
- Hyperparameter tuning (`RandomizedSearchCV` with interactive trials inspection)
- Standalone code (Python) and comprehensive HTML report generation
- Live model serving and interactive playground with uncertainty quantification ($R^2$, MAE error margin, prediction intervals)
- Asynchronous inference logging and real-time data & model drift monitoring (KS-Test & PSI)

Explicitly out of scope:
- Deep learning computer vision or speech modalities
- Autonomous unreviewed data mutation or deletion
- Unrestricted LLM-generated transformation code without validation

## Design principle

**Automate execution, not judgment.**

Statistical and ML engines produce evidence. The user makes consequential choices. An optional LLM layer explains evidence and decisions but must not become the source of truth for numerical/statistical decisions.
