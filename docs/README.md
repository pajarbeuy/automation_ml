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

## MVP scope

The first release targets **tabular classification** with CSV input.

Included:
- dataset profiling
- missing-value analysis
- duplicate detection
- numerical/categorical detection
- outlier detection
- class imbalance analysis
- EDA visualization
- preprocessing recommendations
- human approval/override
- sklearn-compatible pipelines
- baseline model comparison
- cross-validation
- experiment tracking
- model and pipeline export

Explicitly out of MVP:
- autonomous feature engineering
- deep learning
- NLP/CV
- production model serving
- autonomous deployment
- unrestricted LLM-generated transformations

## Design principle

**Automate execution, not judgment.**

Statistical and ML engines produce evidence. The user makes consequential choices. An optional LLM layer explains evidence and decisions but must not become the source of truth for numerical/statistical decisions.
