# System Architecture

## 1. Architectural principle

Use deterministic ML/statistical components for decisions and an optional AI layer for explanation and interaction.

```text
                         ┌──────────────────────┐
                         │      Frontend        │
                         │ React / TypeScript   │
                         └──────────┬───────────┘
                                    │ REST
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │ API + orchestration  │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      ┌────────────┐       ┌────────────────┐      ┌─────────────┐
      │ Profiling  │       │ Recommendation │      │ Experiment  │
      │ Engine     │       │ Engine         │      │ Engine      │
      └─────┬──────┘       └───────┬────────┘      └──────┬──────┘
            │                      │                       │
            └──────────────┬───────┴──────────────┬────────┘
                           ▼                      ▼
                   ┌──────────────┐       ┌──────────────┐
                   │ Pipeline     │       │ Artifact     │
                   │ Builder      │       │ Manager      │
                   └──────┬───────┘       └──────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ scikit-learn  │
                  │ Pipeline      │
                  └───────────────┘
```

## 2. Components

### Frontend

Responsibilities:
- project dashboard
- upload flow
- EDA
- recommendation review
- pipeline builder
- experiment comparison
- artifact export

### API

Responsibilities:
- authentication in future
- project CRUD
- dataset lifecycle
- workflow orchestration
- experiment submission
- result retrieval

### Profiling engine

Pure Python/statistical module.

Input:
- dataframe

Output:
- machine-readable profile JSON

### Recommendation engine

Consumes profile and workflow context.

Output:
- candidate actions
- scores
- rationale
- confidence
- evidence

It must not mutate the dataframe.

### Pipeline builder

Converts approved decisions into executable sklearn components.

Preferred abstraction:
- ColumnTransformer
- Pipeline
- custom transformers (Winsorizer, FrequencyEncoder, AutoFeatureEngineer)

### Experiment & Tuning engine

Runs:
- train/validation
- stratified CV (Classification) and KFold CV (Regression)
- candidate models (linear, tree-based, boosting, KNN, XGBoost, LightGBM)
- hyperparameter tuning via RandomizedSearchCV
- metric calculation ($R^2$, MAE, RMSE, Accuracy, F1, ROC-AUC)
- artifact persistence and MLflow model tracking

### Serving & Inference engine

Responsibilities:
- schema resolution for candidate model features
- single-row live inference and batch CSV processing
- classification probability estimation
- regression uncertainty quantification ($R^2$ variance explained, $\pm\text{MAE}$ margin of error, prediction intervals)
- automatic tolerance for unselected or dropped columns

### Drift Monitoring engine

Responsibilities:
- asynchronous non-blocking prediction logging (`PredictionLog`)
- two-sample Kolmogorov-Smirnov test and Wasserstein distance for continuous features
- Population Stability Index (PSI) for categorical features
- traffic simulation and real-time health reporting

### Artifact & Export manager

Stores and exports:
- serialized Scikit-Learn pipelines (`pipeline.pkl`)
- metrics and cross-validation summaries (`metrics.json`)
- standalone executable Python training script (`train_pipeline.py`)
- standalone comprehensive HTML experiment reports

## 3. Repository structure

```text
ml-autonomus/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── routers/            # projects, datasets, recommendations, pipelines, experiments, serving, monitoring, assistant
│   │   ├── db_models/          # SQLAlchemy ORM (Project, Dataset, WorkflowDecision, Experiment, Metric, Artifact, PredictionLog)
│   │   └── storage/            # Local filesystem artifact & dataset storage
│   └── web/                    # React 18 + TypeScript + Vite frontend
│       ├── src/pages/          # Dashboard, Upload, EDA, Treatment, Pipeline, Experiments, Result, DriftMonitoring
│       └── src/components/     # Sidebar, Modals, ErrorBoundary, AIAssistantWidget
├── packages/
│   └── ml_engine/              # Pure Python ML package
│       ├── profiling/          # Dataset profiler & statistics
│       ├── eda/                # Visual chart data generators
│       ├── detection/          # Missing, outlier, imbalance detectors
│       ├── recommendations/    # Task-aware recommendation engine
│       ├── preprocessing/      # Pipeline builder & custom transformers
│       ├── models/             # Model registry & adapters (sklearn, xgboost, lightgbm)
│       ├── evaluation/         # CV evaluators (StratifiedKFold & KFold)
│       ├── experiments/        # Experiment runner & hyperparameter tuner
│       ├── serving/            # Live inference & uncertainty estimation
│       ├── monitoring/         # Statistical drift detector (KS-test, PSI)
│       ├── assistant/          # Rule-based natural language explainer
│       └── export/             # Standalone Python script & HTML report generators
├── tests/                      # Pytest suite (79 tests)
├── docs/                       # Architectural specs & roadmaps
├── docker-compose.yml
└── pyproject.toml
```

## 4. Execution model

MVP can use synchronous jobs for small datasets.

Future:
- Redis queue
- Celery/RQ
- worker containers
- object storage

## 5. Storage

PostgreSQL:
- projects
- datasets metadata
- workflow decisions
- experiments
- metrics
- artifacts metadata

Object storage:
- raw datasets
- trained models
- serialized pipelines
- generated reports

## 6. Optional LLM layer

```text
Profile JSON
     ↓
Recommendation JSON
     ↓
LLM
     ↓
Human-readable explanation
```

The LLM should not:
- calculate statistics
- decide thresholds autonomously
- modify pipelines directly
- invent evidence

## 7. Security principles

- validate uploaded files
- limit file size
- isolate execution
- never execute arbitrary uploaded code
- sanitize filenames
- use project-scoped artifact access
- never expose environment secrets to ML jobs
