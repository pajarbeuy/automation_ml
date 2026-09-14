# Backlog

## P0 (Completed)

- [x] CSV upload
- [x] target selection
- [x] dataset profiling
- [x] EDA visualization workspace
- [x] missing detection
- [x] outlier detection
- [x] categorical/numerical detection
- [x] recommendation cards
- [x] human approve/reject/override
- [x] sklearn pipeline generation
- [x] baseline classification
- [x] CV evaluation (StratifiedKFold)
- [x] experiment comparison
- [x] export pipeline (`pipeline.pkl`)

## P1 (Completed)

- [x] MLflow tracking & model registration
- [x] regression task support (metrics $R^2$, MAE, MSE, RMSE, residual analysis)
- [x] hyperparameter optimization (`RandomizedSearchCV` + tuning modal)
- [x] generated standalone Python code (`train_pipeline.py`)
- [x] project history & experiment comparison
- [x] comprehensive HTML report generation
- [x] asynchronous inference logging (`PredictionLog`)

## P2 (In Progress & Extensions)

- [x] Rule-based AI assistant & natural language explainer
- [x] automated feature engineering (`AutoFeatureEngineer` log1p skew & interactions)
- [x] XGBoost & LightGBM native adapters
- [x] model serving & live prediction playground with uncertainty quantification
- [x] native data & model drift monitoring (KS-Test, PSI)
- [ ] learned recommendation ranking from historical runs
- [ ] distributed async job workers (Celery / Redis)
- [ ] multi-tenant authentication & access control
- [ ] cloud object storage (S3 / MinIO)

## Research backlog

- Does recommendation assistance reduce workflow time?
- Does it improve preprocessing consistency?
- How often do users override recommendations?
- Which evidence signals correlate with useful recommendations?
- Can historical experiments train a recommendation model?
- How should recommendation uncertainty be calibrated?
