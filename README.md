# AutoML Human-in-the-Loop

> From raw CSV to validated ML pipeline — with human-controlled preprocessing decisions.

## Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.12, FastAPI, SQLAlchemy, Pydantic v2 |
| ML Engine | scikit-learn, pandas, numpy, scipy |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS v4 |
| Database | PostgreSQL 16 |
| Infrastructure | Docker Compose (WSL Ubuntu) |

## Quick Start (WSL)

```bash
# From WSL, navigate to project
cd /mnt/d/ml-autonomus

# Start all services
docker-compose up --build

# Open in browser
http://localhost:5173
```

## Development (without Docker)

### Backend

```bash
cd apps/api
pip install -e "../../[dev]"
# Set DATABASE_URL env var to a local PostgreSQL
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd apps/web
npm run dev
```

## Run Tests

```bash
# From project root
pip install -e ".[dev]"
pytest tests/ -v
```

## User Journey

1. **Dashboard** → Create project
2. **Upload** → Drop CSV, select target column → automatic profiling
3. **EDA** → Inspect distributions, missing values, class balance
4. **Treatment** → Review recommendations, accept/override/reject
5. **Pipeline** → Generate leakage-safe sklearn Pipeline, validate
6. **Experiments** → Train multiple classifiers, compare with CV

## Architecture

```
apps/api/          FastAPI backend
apps/web/          React + Vite frontend
packages/
  ml_engine/
    profiling/     Dataset profiler
    eda/           EDA chart data builder
    detection/     Missing / outlier / imbalance detectors
    recommendations/ Evidence-based recommendation engine
    preprocessing/ Pipeline builder + custom transformers
    models/        Model registry
    evaluation/    Stratified CV evaluator
    experiments/   End-to-end experiment runner
tests/             pytest test suite
docker-compose.yml
```

## Safety Principles

- Raw datasets are **immutable** — never modified in place
- Preprocessing transformations are **fitted only on training data** (leakage-safe)
- Every recommendation exposes **evidence** and **alternatives**
- Anomaly detection ≠ automatic deletion — all treatments require human approval
- Experiment results include **random seed + dependency versions** for reproducibility
