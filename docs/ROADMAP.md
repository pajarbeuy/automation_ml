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

## Phase 8: Production

Potential:
- async workers
- object storage
- authentication
- multi-user projects
- model registry
- model serving
- monitoring
