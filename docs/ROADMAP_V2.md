# AutoML Platform --- Product Roadmap

> An evidence-driven, task-aware AutoML platform for dataset profiling,
> preprocessing recommendation, pipeline generation, experimentation,
> model serving, monitoring, and AI-assisted workflow explanation.

------------------------------------------------------------------------

# 0. Product Principles

The platform must follow these principles throughout all phases.

## 0.1 Evidence Before Recommendation

The system must never recommend a treatment without measurable evidence.

Bad:

``` text
Dataset contains numerical features
→ use StandardScaler
```

Good:

``` text
Model family: KNN
Feature scale variance detected
→ StandardScaler recommended
```

------------------------------------------------------------------------

## 0.2 Task-Aware Processing

Every recommendation must understand the machine learning task.

Supported tasks:

-   Classification
-   Regression
-   Time Series Regression / Forecasting

A treatment that is not applicable to a task must not be presented as an
actionable recommendation.

Example:

``` text
Regression
→ Class imbalance
→ NOT_APPLICABLE
```

Not:

``` text
Regression
→ Class imbalance
→ SMOTE
```

------------------------------------------------------------------------

## 0.3 Model-Aware Processing

Preprocessing decisions must consider the downstream model.

Example:

``` text
KNN
→ scaling recommended

Random Forest
→ scaling usually unnecessary
```

The system must not assume that one preprocessing strategy is optimal
for every model.

------------------------------------------------------------------------

## 0.4 Leakage Safety

All transformations that learn from data must be fitted only on training
data.

Examples:

-   imputation
-   scaling
-   encoding
-   feature selection
-   target transformation
-   oversampling

Use sklearn Pipeline / ColumnTransformer whenever possible.

------------------------------------------------------------------------

## 0.5 Candidate-Based Optimization

The system should generate candidate treatments rather than blindly
apply every detected issue.

Example:

``` text
Target skewness detected
        ↓
Generate candidates
        ↓
Original target
Log1p target
Yeo-Johnson target
        ↓
Cross-validation
        ↓
Compare performance
        ↓
Select best candidate
```

------------------------------------------------------------------------

## 0.6 Explainability

Every recommendation must have:

-   evidence
-   reasoning
-   expected effect
-   potential risk
-   validation result

The AI assistant explains structured evidence.

It must not invent statistical conclusions.

------------------------------------------------------------------------

# Phase 0 --- Foundation

## Goal

Establish a stable application architecture before implementing machine
learning intelligence.

## Backend

-   [x] Repository structure
-   [x] Python package structure
-   [x] FastAPI application
-   [x] API versioning
-   [x] Configuration management
-   [x] Environment variable management
-   [x] Error handling
-   [x] Logging
-   [x] Request validation
-   [x] Response schemas

## Frontend

-   [x] React application
-   [x] Routing
-   [x] Layout
-   [x] Project workspace
-   [x] Global state management
-   [x] API client
-   [x] Loading states
-   [x] Error states

## Infrastructure

-   [x] Docker Compose
-   [x] API Dockerfile
-   [x] Web Dockerfile
-   [x] Development environment
-   [x] Production environment configuration

## Testing

-   [x] Unit test structure
-   [x] API tests
-   [x] Basic frontend tests
-   [x] ML utility tests

## Dataset Input

-   [x] CSV upload
-   [x] File validation
-   [x] File size validation
-   [x] Encoding detection
-   [x] Dataset storage
-   [x] Dataset metadata

------------------------------------------------------------------------

# Phase 1 --- Dataset Profiling + EDA

## Goal

Understand the dataset before making any treatment recommendation.

## 1.1 Schema Detection

-   [x] Column names
-   [x] Data types
-   [x] Numerical columns
-   [x] Categorical columns
-   [x] Boolean columns
-   [x] Datetime columns
-   [x] Text columns
-   [x] Mixed-type detection
-   [x] Unsupported-type detection

## 1.2 Task Detection

Detect or request confirmation of:

-   [x] Classification
-   [x] Regression
-   [ ] Time-series

Task detection must consider:

-   target dtype
-   number of unique target values
-   target cardinality
-   target distribution
-   datetime presence
-   temporal ordering

The system must allow the user to override automatic task detection.

## 1.3 Target Validation

-   [x] Target existence
-   [x] Target dtype validation
-   [x] Target missing values
-   [x] Target cardinality
-   [x] Target constant-value detection
-   [ ] Target leakage detection

### Classification Target

-   [x] Number of classes
-   [x] Class frequencies
-   [x] Class proportions
-   [x] Minority class
-   [x] Imbalance ratio

### Regression Target

-   [x] Mean
-   [x] Median
-   [x] Standard deviation
-   [x] Min
-   [x] Max
-   [x] Quantiles
-   [x] Skewness
-   [x] Kurtosis
-   [x] Outlier ratio

## 1.4 Data Quality

-   [x] Missing values
-   [x] Missing percentage
-   [x] Duplicate rows
-   [x] Constant features
-   [x] Near-constant features
-   [x] High-cardinality categorical features
-   [x] Invalid values
-   [ ] Impossible values
-   [ ] Potential leakage
-   [ ] ID-like columns

## 1.5 Numerical Analysis

-   [x] Descriptive statistics
-   [x] Distribution
-   [x] Skewness
-   [x] Kurtosis
-   [x] Outlier detection
-   [x] IQR
-   [x] Quantiles
-   [x] Variance
-   [x] Zero variance

## 1.6 Categorical Analysis

-   [x] Unique values
-   [x] Cardinality
-   [x] Frequency distribution
-   [x] Rare categories
-   [x] Missing category
-   [x] High-cardinality detection

## 1.7 Relationship Analysis

### Classification

-   [x] Feature vs target distribution
-   [ ] Mutual information
-   [x] Numerical feature analysis
-   [x] Categorical feature analysis
-   [x] Correlation analysis

### Regression

-   [x] Feature vs target relationship
-   [x] Pearson correlation
-   [ ] Spearman correlation
-   [ ] Mutual information
-   [ ] Non-linear relationship detection

## 1.8 Time-Series Detection

-   [ ] Datetime column detection
-   [ ] Temporal ordering
-   [ ] Frequency detection
-   [ ] Duplicate timestamps
-   [ ] Missing periods
-   [ ] Trend detection
-   [ ] Seasonality candidates

## 1.9 Visualization

-   [x] Numerical distributions
-   [x] Box plots
-   [x] Histograms
-   [x] Categorical frequency charts
-   [x] Correlation heatmap
-   [x] Target distribution
-   [x] Target vs feature
-   [ ] Time-series charts

## 1.10 Profile Persistence

-   [x] Profile schema
-   [x] Profile versioning
-   [x] Dataset fingerprint
-   [x] Profile persistence
-   [x] Profile retrieval

------------------------------------------------------------------------

# Phase 2 --- Recommendation Engine

## Goal

Convert profiling evidence into safe, task-aware treatment candidates.

## 2.1 Recommendation Architecture

Every rule should contain:

``` python
Rule(
    name="class_imbalance",
    applies_to=["classification"],
    condition=detect_class_imbalance,
    recommendation="class_weight",
    evidence=...,
    severity=...,
)
```

Each recommendation should contain:

``` text
name
category
task
applicability
severity
confidence
evidence
reason
recommended_action
alternatives
risks
validation_required
```

## 2.2 Recommendation States

-   [x] DETECTED
-   [x] RECOMMENDED
-   [x] ACCEPTED
-   [x] REJECTED
-   [x] OVERRIDDEN
-   [ ] NOT_APPLICABLE
-   [ ] VALIDATED

## 2.3 Universal Recommendations

Applicable to both classification and regression where appropriate.

### Missing Values

-   [x] Mean imputation
-   [x] Median imputation
-   [x] Mode imputation
-   [x] Constant category
-   [ ] Missing indicator
-   [ ] Advanced imputation candidates

### Duplicates

-   [x] Exact duplicate detection
-   [x] Duplicate removal recommendation
-   [ ] Duplicate investigation

### Constant Features

-   [x] Constant feature removal
-   [x] Near-constant feature warning

### Categorical Encoding

-   [x] One-hot encoding
-   [x] Ordinal encoding
-   [ ] Frequency encoding
-   [x] High-cardinality warning

### Scaling

-   [x] StandardScaler
-   [x] MinMaxScaler
-   [x] RobustScaler
-   [x] No scaling

Scaling must depend on:

-   feature distribution
-   outliers
-   model family

### Outliers

-   [x] IQR detection
-   [x] Z-score detection
-   [x] Robust statistics
-   [x] Outlier warning
-   [x] Winsorization candidate
-   [ ] Transformation candidate

The system must never automatically delete all outliers.

## 2.4 Classification Recommendations

### Class Imbalance

-   [x] Class distribution analysis
-   [x] Imbalance ratio
-   [x] Minority class detection
-   [x] Severity classification

Candidates:

-   [x] `class_weight="balanced"`
-   [x] SMOTE
-   [x] Random oversampling
-   [x] Random undersampling
-   [ ] Combination methods

### Splitting

-   [x] Stratified train/test split
-   [x] StratifiedKFold

### Threshold

-   [x] Default threshold analysis
-   [ ] Precision/Recall tradeoff
-   [ ] Threshold optimization candidate

### Metrics

-   [x] Accuracy
-   [x] Precision
-   [x] Recall
-   [x] F1
-   [x] ROC-AUC
-   [x] PR-AUC
-   [x] Confusion matrix

## 2.5 Regression Recommendations

### Target Distribution

-   [x] Target skewness
-   [x] Target kurtosis
-   [x] Target range
-   [x] Target quantiles

### Target Transformation

Candidates:

-   [x] No transformation (`keep_original`)
-   [x] Log1p (`log1p_transform`)
-   [x] Yeo-Johnson (`yeo_johnson_transform`)
-   [ ] Power transformation

Transformations must be evaluated through cross-validation.

### Target Outliers

-   [x] Target outlier detection
-   [x] Extreme-value analysis
-   [x] Outlier warning
-   [x] Robust modeling candidate (`use_robust_models`)
-   [x] Transformation candidate (`clip_target_iqr`)

The system must not automatically delete target outliers.

### Metrics

-   [x] MAE
-   [x] MSE
-   [x] RMSE
-   [x] R²
-   [ ] MAPE where appropriate
-   [ ] sMAPE where appropriate

## 2.6 Time-Series Recommendations

-   [ ] Temporal split
-   [ ] TimeSeriesSplit
-   [ ] Lag features
-   [ ] Rolling statistics
-   [ ] Date features
-   [ ] Trend features
-   [ ] Seasonality features
-   [ ] Future-data leakage detection

## 2.7 Recommendation UI

-   [x] Recommendation cards
-   [x] Evidence display
-   [x] Severity
-   [x] Confidence
-   [x] Reasoning
-   [x] Accept
-   [x] Reject
-   [x] Override
-   [x] Alternative treatment
-   [ ] Not Applicable explanation
-   [x] Recommendation history

------------------------------------------------------------------------

# Phase 3 --- Pipeline Engine

## Goal

Convert accepted recommendations into reproducible, leakage-safe ML
pipelines.

## 3.1 Pipeline Construction

-   [x] ColumnTransformer
-   [x] sklearn Pipeline
-   [x] Pipeline configuration
-   [x] Pipeline validation
-   [x] Pipeline serialization
-   [x] Pipeline deserialization

## 3.2 Numeric Pipeline

Possible stages:

``` text
Missing Value Imputation
        ↓
Outlier / Transformation
        ↓
Scaling
```

## 3.3 Categorical Pipeline

Possible stages:

``` text
Missing Value Imputation
        ↓
Rare Category Handling
        ↓
Encoding
```

## 3.4 Classification Pipeline

Support:

-   [x] preprocessing
-   [x] class weighting
-   [ ] sampling
-   [x] classifier
-   [ ] probability calibration where appropriate
-   [ ] threshold configuration

## 3.5 Regression Pipeline

Support:

-   [x] preprocessing
-   [ ] target transformation
-   [x] regressor
-   [ ] inverse transformation

## 3.6 Time-Series Pipeline

Support:

-   [ ] temporal feature engineering
-   [ ] lag features
-   [ ] rolling features
-   [ ] temporal splitting
-   [ ] forecasting-safe transformations

## 3.7 Leakage Protection

-   [x] Fit preprocessing only on training data
-   [x] Prevent test-set statistics from entering preprocessing
-   [ ] Prevent future data leakage
-   [ ] Prevent target leakage
-   [x] Validate pipeline graph

## 3.8 Pipeline Preview

Display:

``` text
Input
 ↓
Feature Detection
 ↓
Imputation
 ↓
Encoding
 ↓
Scaling
 ↓
Feature Engineering
 ↓
Model
 ↓
Prediction
```

------------------------------------------------------------------------

# Phase 4 --- Experiment Engine

## Goal

Automatically evaluate candidate pipelines and models.

## 4.1 Dataset Splitting

### Classification

-   [x] Train/test split
-   [x] Stratified split

### Regression

-   [x] Train/test split
-   [x] KFold
-   [ ] RepeatedKFold
-   [ ] Optional target-binned stratification

### Time-Series

-   [ ] Temporal train/test split
-   [ ] TimeSeriesSplit

## 4.2 Baseline Models

### Classification

-   [x] Logistic Regression
-   [x] Decision Tree
-   [x] Random Forest
-   [x] Gradient Boosting
-   [x] HistGradientBoosting
-   [x] KNN
-   [x] XGBoost
-   [x] LightGBM

### Regression

-   [x] Linear Regression
-   [x] Ridge
-   [x] Random Forest Regressor
-   [x] Gradient Boosting Regressor
-   [x] HistGradientBoosting Regressor
-   [x] KNN Regressor
-   [x] XGBRegressor
-   [x] LGBMRegressor

## 4.3 Cross Validation

The CV strategy must be selected automatically from dataset context.

``` text
Classification
→ StratifiedKFold

Regression
→ KFold / RepeatedKFold

Time Series
→ TimeSeriesSplit
```

Never hard-code one CV strategy for every task.

## 4.4 Hyperparameter Optimization

-   [x] RandomizedSearchCV
-   [ ] GridSearchCV where search space is small
-   [x] Task-aware search spaces
-   [x] Model-specific search spaces
-   [ ] Search budget
-   [ ] Early stopping where supported
-   [x] Reproducible random seeds

## 4.5 Experiment Metrics

### Classification

-   [x] Accuracy
-   [x] Precision
-   [x] Recall
-   [x] F1
-   [x] ROC-AUC
-   [x] PR-AUC
-   [x] Confusion matrix

### Regression

-   [x] R²
-   [x] MAE
-   [x] MSE
-   [x] RMSE
-   [ ] MAPE
-   [ ] sMAPE
-   [x] Actual vs Predicted
-   [x] Residual Plot

## 4.6 Experiment Comparison

Compare:

``` text
Model
Pipeline
Features
Transformations
CV strategy
Hyperparameters
Metrics
Training time
Inference time
```

## 4.7 Experiment Persistence

-   [x] Dataset version
-   [x] Pipeline version
-   [x] Model version
-   [x] Hyperparameters
-   [x] Metrics
-   [x] CV results
-   [x] Random seed
-   [x] Environment information
-   [x] Experiment timestamp

------------------------------------------------------------------------

# Phase 5 --- Export + Experiment Tracking

## Goal

Make trained ML systems reproducible and portable.

## 5.1 Model Export

-   [x] Pickle
-   [x] Joblib
-   [x] MLflow model
-   [x] Model metadata

## 5.2 Pipeline Export

-   [x] Full preprocessing pipeline
-   [x] Model
-   [ ] Target transformer
-   [x] Feature schema

## 5.3 Configuration Export

Export:

``` json
{
  "task": "regression",
  "target": "price",
  "features": [],
  "preprocessing": {},
  "feature_engineering": {},
  "model": {},
  "evaluation": {}
}
```

## 5.4 Report Generation

-   [ ] Dataset profile report
-   [ ] Treatment report
-   [ ] Pipeline report
-   [ ] Experiment report
-   [ ] Model evaluation report
-   [ ] Final model report

## 5.5 MLflow

-   [x] MLflow integration
-   [x] Experiment tracking
-   [x] Parameter logging
-   [x] Metric logging
-   [x] Artifact logging
-   [x] Model logging
-   [x] Dataset metadata
-   [x] Model versioning

------------------------------------------------------------------------

# Phase 6 --- AI Assistant

## Goal

Provide natural-language interaction over deterministic ML evidence.

## 6.1 Explanation

-   [x] "Why was this recommended?"
-   [x] "Why was this rejected?"
-   [x] "Why was this model selected?"
-   [x] "Why is this feature being transformed?"
-   [x] "Why is scaling unnecessary?"

## 6.2 Experiment Explanation

Examples:

``` text
Why did Random Forest win?

Why did Log1p improve the regression?

Why did the F1 score decrease?

Why is recall important for this dataset?
```

## 6.3 Workflow Assistant

-   [x] Explain current workflow
-   [x] Explain next step
-   [x] Navigate to relevant workspace
-   [x] Explain pipeline
-   [x] Explain experiment results

## 6.4 Evidence Contract

The AI receives structured evidence:

``` json
{
  "task": "regression",
  "target": "price",
  "target_skewness": 2.84,
  "candidate_transformations": [
    "none",
    "log1p",
    "yeo_johnson"
  ],
  "cv_results": {
    "none": 0.81,
    "log1p": 0.87,
    "yeo_johnson": 0.84
  }
}
```

The AI explains the evidence.

It must not generate statistical values unless those values actually
exist in the structured evidence.

------------------------------------------------------------------------

# Phase 7 --- Advanced AutoML

## Goal

Move from automated preprocessing toward automated pipeline search and
optimization.

## 7.1 Automated Feature Engineering

-   [x] `AutoFeatureEngineer`
-   [x] Log1p skew candidates
-   [x] Numeric interactions
-   [ ] Task-aware feature engineering
-   [ ] Polynomial candidates
-   [ ] Ratio features
-   [ ] Date/time features
-   [ ] Group statistics
-   [ ] Feature selection
-   [ ] Feature importance analysis

Feature engineering must be validated through CV.

## 7.2 Automated Treatment Search

Instead of:

``` text
detect issue
→ apply treatment
```

implement:

``` text
detect issue
→ generate candidates
→ construct candidate pipelines
→ evaluate
→ rank
→ select
```

Example:

``` text
Missing Values
├── Median
├── Mean
└── Model-based

Scaling
├── None
├── StandardScaler
└── RobustScaler

Target
├── None
├── Log1p
└── Yeo-Johnson
```

Then search combinations within a controlled budget.

## 7.3 Model Selection

-   [ ] Automated model ranking
-   [ ] Model family selection
-   [ ] Pipeline + model comparison
-   [ ] Training-time constraint
-   [ ] Memory constraint
-   [ ] Inference-time constraint

## 7.4 Advanced HPO

-   [x] RandomizedSearchCV
-   [ ] Bayesian optimization
-   [ ] Optuna integration
-   [ ] Conditional search spaces
-   [ ] Multi-objective optimization

Potential objectives:

``` text
maximize performance
minimize training time
minimize inference latency
minimize model complexity
```

## 7.5 Ensemble Models

-   [ ] Voting
-   [ ] Bagging
-   [ ] Stacking
-   [ ] Blending

Only enable when justified by experiment results.

------------------------------------------------------------------------

# Phase 8 --- Production

## Goal

Deploy trained models as reliable production services.

## 8.1 Model Serving

-   [x] `POST /api/projects/{id}/predict`
-   [x] Raw JSON prediction
-   [x] Candidate feature schema
-   [x] Batch prediction
-   [x] Request validation
-   [x] Schema validation
-   [ ] Prediction versioning
-   [ ] Model version selection

## 8.2 Prediction Playground

-   [x] Single prediction
-   [x] Batch CSV
-   [x] API snippets
-   [x] Input validation
-   [ ] Prediction history
-   [ ] Prediction metadata

## 8.3 Containerization

-   [x] API Dockerfile
-   [x] Web Dockerfile
-   [x] Nginx
-   [x] Docker Compose
-   [ ] Production compose profile
-   [ ] Health checks
-   [ ] Resource limits

## 8.4 Model Registry

-   [x] MLflow model logging
-   [ ] Model versions
-   [ ] Model stages
-   [ ] Model promotion
-   [ ] Model rollback
-   [ ] Champion / challenger models

## 8.5 Monitoring

### Data Drift

-   [x] KS-Test
-   [x] PSI
-   [x] Background drift logging
-   [x] Drift simulator
-   [x] Drift dashboard

### Data Quality

-   [ ] Missing-value monitoring
-   [ ] Invalid-value monitoring
-   [ ] Feature schema changes
-   [ ] Distribution monitoring

### Prediction Monitoring

-   [ ] Prediction distribution
-   [ ] Prediction anomalies
-   [ ] Prediction volume
-   [ ] Latency

### Model Performance

-   [ ] Delayed ground-truth ingestion
-   [ ] Performance degradation
-   [ ] Classification metric monitoring
-   [ ] Regression metric monitoring

### Alerting

-   [x] Drift threshold
-   [ ] Data quality threshold
-   [ ] Performance threshold
-   [ ] Notification system

## 8.6 Async Processing

-   [ ] Redis
-   [ ] Celery
-   [ ] Background dataset profiling
-   [ ] Background experiments
-   [ ] Background HPO
-   [ ] Background report generation
-   [ ] Job status tracking
-   [ ] Retry mechanism

Architecture:

``` text
Frontend
   ↓
FastAPI
   ↓
Job Queue
   ↓
Worker
   ↓
ML Pipeline
   ↓
Experiment / Artifact Storage
```

## 8.7 Authentication

-   [ ] User registration
-   [ ] Login
-   [ ] JWT / session
-   [ ] User roles
-   [ ] Project ownership
-   [ ] Multi-user projects
-   [ ] Access control

## 8.8 Storage

-   [ ] Local artifact storage
-   [ ] MinIO
-   [ ] S3-compatible storage
-   [ ] Dataset versioning
-   [ ] Model artifact storage
-   [ ] Experiment artifact storage

------------------------------------------------------------------------

# Phase 9 --- Reliability + MLOps

## Goal

Turn the platform from a functional AutoML application into a
production-grade ML platform.

## 9.1 Reproducibility

-   [ ] Dataset fingerprint
-   [ ] Dataset version
-   [ ] Pipeline version
-   [ ] Model version
-   [ ] Dependency lockfile
-   [ ] Random seed
-   [ ] Environment metadata

## 9.2 Model Lifecycle

``` text
Experiment
    ↓
Candidate
    ↓
Validated
    ↓
Registered
    ↓
Staging
    ↓
Production
    ↓
Monitoring
    ↓
Retraining
```

Implement:

-   [ ] Model approval
-   [ ] Model promotion
-   [ ] Rollback
-   [ ] Retirement

## 9.3 Automated Retraining

-   [ ] Drift trigger
-   [ ] Performance trigger
-   [ ] Scheduled retraining
-   [ ] Dataset refresh
-   [ ] Automatic experiment
-   [ ] Candidate comparison
-   [ ] Approval gate
-   [ ] Model promotion

Never automatically replace production models without a validation gate.

## 9.4 Observability

-   [ ] Application logs
-   [ ] Worker logs
-   [ ] API latency
-   [ ] Error rate
-   [ ] Job duration
-   [ ] Model inference latency
-   [ ] Resource utilization
-   [ ] Metrics dashboard

------------------------------------------------------------------------

# Phase 10 --- Advanced Intelligence

## Goal

Make the system capable of intelligently searching for good ML solutions
while keeping decisions auditable.

## 10.1 Dataset-Aware Search

The search space should adapt based on:

``` text
dataset size
feature count
feature types
task
cardinality
missingness
outliers
target distribution
time-series properties
compute budget
```

## 10.2 Intelligent Search Budget

Small dataset:

``` text
more candidate pipelines
```

Large dataset:

``` text
fewer expensive candidates
```

The system should estimate:

-   [ ] dataset complexity
-   [ ] training cost
-   [ ] memory requirements
-   [ ] expected search duration

## 10.3 Meta-Learning

Future capability:

-   [ ] Store historical experiments
-   [ ] Learn dataset characteristics
-   [ ] Recommend promising model families
-   [ ] Recommend promising preprocessing strategies
-   [ ] Warm-start HPO

## 10.4 Explainable AutoML Decisions

Every final model should provide:

``` text
Dataset
↓
Detected characteristics
↓
Candidate treatments
↓
Candidate pipelines
↓
Experiments
↓
CV results
↓
Selected model
↓
Reason for selection
```

------------------------------------------------------------------------

# Cross-Phase Architecture

The platform should ultimately follow this architecture:

``` text
                    DATASET
                       │
                       ▼
                ┌─────────────┐
                │   PROFILER  │
                └──────┬──────┘
                       │
                       ▼
              ┌─────────────────┐
              │ DATASET CONTEXT │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   Classification  Regression   Time Series
          │            │            │
          └────────────┼────────────┘
                       ▼
              ┌────────────────┐
              │ RULE ENGINE    │
              │ Applicability  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ EVIDENCE ENGINE │
              └───────┬────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ RECOMMENDATIONS  │
             └────────┬─────────┘
                      │
               Accept / Reject
                      │
                      ▼
             ┌──────────────────┐
             │ PIPELINE ENGINE  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ EXPERIMENT ENGINE│
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ MODEL SELECTION  │
             └────────┬─────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     MODEL REGISTRY          REPORT / AI
          │
          ▼
       SERVING
          │
          ▼
     MONITORING
          │
          ▼
       RETRAINING
          │
          └──────────────► EXPERIMENT
```

------------------------------------------------------------------------

# Current Project Status

## Completed

### Phase 0 --- Foundation
-   [x] Full-stack architecture (FastAPI backend, React frontend, SQLite DB, Docker Compose)
-   [x] Dataset upload, validation, encoding detection, and local storage
-   [x] Automated test suite (pytest + Vite/TS compiler)

### Phase 1 --- Dataset Profiling + EDA
-   [x] Type inference, missing values, duplicates, cardinality, and constant detection
-   [x] Numerical statistics (min, max, mean, std, skewness, kurtosis, IQR)
-   [x] Categorical frequencies and relationship analysis (Pearson correlation, feature vs target)
-   [x] Interactive visualizations (Histograms, boxplots, correlation heatmap, target distribution)
-   [x] Dataset fingerprinting & persistence

### Phase 2 --- Recommendation Engine
-   [x] Evidence-based preprocessing recommendations (imputation, scaling, encoding, outlier clipping)
-   [x] Human-in-the-Loop decision workflow (Accept, Reject, Override) with persistence

### Phase 3 --- Pipeline Engine
-   [x] Leakage-safe `sklearn` ColumnTransformer & Pipeline generation
-   [x] Pipeline configuration validation and serialization (`pipeline.pkl`)

### Phase 4 --- Experiment Engine
-   [x] Baseline models for classification and regression
-   [x] Automated cross-validation (StratifiedKFold for classification, KFold for regression)
-   [x] Comprehensive evaluation metrics & visual evaluation (Actual vs Predicted, Residual Plot)

### Phase 5 --- Export & Tracking
-   [x] Model & pipeline artifact persistence (`pipeline.pkl`, metadata JSON)
-   [x] Full MLflow Tracking & Model Registry integration

### Phase 6 --- AI Assistant
-   [x] Evidence-grounded natural-language workflow explainer widget
-   [x] Contextual explanations for recommendations, rejections, and experiment results

### Phase 7 --- Advanced AutoML
-   [x] Automated Feature Engineering (`AutoFeatureEngineer`)
-   [x] Log1p skew candidate & numeric interaction features
-   [x] RandomizedSearchCV with model-specific hyperparameter search spaces
-   [x] Regression model support (`LinearRegression`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `KNNRegressor`, `XGBRegressor`, `LGBMRegressor`)
-   [x] Regression metrics ($R^2$, MAE, MSE, RMSE) & Actual vs Predicted / Residual Plots
-   [x] XGBoost & LightGBM native adapters

### Phase 8 --- Production
-   [x] Live model serving (`POST /api/projects/{id}/predict`) with robust schema & input validation
-   [x] Interactive Prediction Playground (single prediction & batch CSV evaluation)
-   [x] Model confidence & uncertainty quantification ($R^2$ variance explained, $\pm\text{MAE}$ margin of error, prediction interval estimation for regression)
-   [x] Docker Compose containerization
-   [x] MLflow model registration
-   [x] Native Model & Data Drift Monitoring Engine (Kolmogorov-Smirnov test, PSI, background logging, drift simulator, real-time dashboard)

------------------------------------------------------------------------

# Immediate Priority

The next development priority should NOT be adding more models.

Prioritize the intelligence and correctness layer:

1.  [x] Dataset Context
2.  [x] Task Detection
3.  [x] Rule Applicability
4.  [x] Classification / Regression separation
5.  [x] `NOT_APPLICABLE` recommendation state (class imbalance omitted for regression; regression target skewness/outliers omitted for classification)
6.  [x] Evidence-based recommendation schema
7.  [x] Task-aware CV selection
8.  [x] Regression target analysis
9.  [ ] Leakage detection
10. [ ] Candidate treatment evaluation
11. [ ] Recommendation validation through experiments

After these are stable:

12. [ ] Async Workers
13. [ ] Object Storage
14. [ ] Authentication
15. [ ] Production Model Lifecycle
16. [ ] Automated Retraining
17. [ ] Advanced AutoML Search

------------------------------------------------------------------------

# Definition of "Autonomous"

The system should not mean:

> "The system automatically does everything."

Instead:

> "The system automatically analyzes the dataset, generates applicable
> treatment candidates, evaluates them objectively, and selects the best
> validated pipeline within defined constraints."

The system must remain:

-   evidence-driven
-   task-aware
-   model-aware
-   leakage-safe
-   reproducible
-   auditable
-   explainable
