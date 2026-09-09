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
- custom transformers where necessary

### Experiment engine

Runs:
- train/validation
- cross-validation
- candidate models
- metric calculation
- artifact persistence

### Artifact manager

Stores:
- model
- pipeline
- configuration
- metrics
- reports

## 3. Suggested repository

```text
automl-hil/
├── apps/
│   ├── api/
│   └── web/
├── packages/
│   └── ml_engine/
│       ├── profiling/
│       ├── eda/
│       ├── detection/
│       ├── recommendations/
│       ├── preprocessing/
│       ├── models/
│       ├── evaluation/
│       └── experiments/
├── tests/
├── docs/
├── scripts/
├── docker/
├── pyproject.toml
├── docker-compose.yml
└── README.md
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
