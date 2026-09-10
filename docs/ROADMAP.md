# Product Roadmap

## Phase 0: Foundation

- repository structure
- Python package
- FastAPI skeleton
- React skeleton
- Docker Compose
- basic tests
- CSV upload

## Phase 1: Profiling + EDA

- schema detection
- missing values
- duplicates
- cardinality
- numerical statistics
- class distribution
- charts
- profile persistence

## Phase 2: Recommendation Engine

- missing-value recommendations
- outlier detection
- encoding recommendations
- scaling recommendations
- recommendation UI
- approve/reject/override

## Phase 3: Pipeline Engine

- ColumnTransformer
- sklearn Pipeline
- leakage-safe transformations
- pipeline preview
- pipeline serialization

## Phase 4: Experiment Engine

- baseline models
- stratified CV
- metrics
- comparison UI
- experiment persistence

## Phase 5: Export + Tracking

- model export
- pipeline export
- configuration export
- report generation
- MLflow integration

## Phase 6: AI Assistant

- natural-language explanations
- "why was this recommended?"
- experiment summaries
- workflow navigation

The AI layer must consume structured evidence rather than inventing statistical conclusions.

## Phase 7: Advanced AutoML (Completed)

- [x] Automated Feature Engineering (`AutoFeatureEngineer`: log1p skew & numeric interactions)
- [x] Hyperparameter Optimization (`RandomizedSearchCV` + KFold / Stratified CV)
- [x] Regression Model Support (`LinearRegression`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `KNNRegressor`, `XGBRegressor`, `LGBMRegressor`)
- [x] Regression Metrics (`R²`, `MAE`, `MSE`, `RMSE`) & Actual vs Predicted Scatter / Residual Plot UI
- [x] Advanced Model Adapters (XGBoost & LightGBM)

## Phase 8: Production (In Progress)

- [x] Model Serving & Live Prediction Engine (`POST /api/projects/{id}/predict` on raw JSON)
- [x] Candidate Feature Schema Generator (`GET /api/projects/{id}/models/candidate/schema`)
- [x] Live Prediction & Inference Playground UI in `ResultPage.tsx` (Single prediction, Batch CSV, API snippets)
- [x] Containerization & Orchestration (`apps/api/Dockerfile`, `apps/web/Dockerfile`, `apps/web/nginx.conf`, `docker-compose.yml`)
- [x] Model Registry Integration (`mlflow.sklearn.log_model`)
- [x] Model Drift Monitoring (`scipy.stats` KS-Test & PSI drift detector, background logging, simulator, dan Neumorphic UI dashboard)
- [ ] Async Workers (Redis / Celery)
- [ ] Authentication & Multi-User Projects
- [ ] Cloud Object Storage (S3 / MinIO)
