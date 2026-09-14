# Product Requirements Document

## 1. Product

**Working name:** AutoML Human-in-the-Loop

**Category:** ML workflow automation / assisted AutoML

**Primary domain:** Tabular machine learning

**Supported tasks:** Classification and Regression

## 2. Problem statement

Building a tabular ML model repeatedly involves the same operational work:

- loading data
- profiling columns
- checking missing values
- detecting duplicates
- inspecting distributions
- identifying outliers
- encoding categorical variables
- scaling numerical variables
- checking class imbalance
- selecting candidate models
- validating models
- comparing experiments
- documenting preprocessing decisions

These tasks are repetitive, but blindly automating them is dangerous because preprocessing decisions depend on dataset context and domain meaning.

The product therefore separates:

**Automation**
- profiling
- detection
- visualization
- candidate generation
- pipeline construction
- training
- evaluation
- experiment tracking

from:

**Human judgment**
- interpreting suspicious data
- selecting among materially different treatments
- defining the optimization objective
- accepting/rejecting recommendations
- deciding whether an observed anomaly is a real phenomenon or bad data

## 3. Product vision

Create a workspace where a user can move from raw tabular data to a validated, reproducible ML pipeline without manually rewriting the same preprocessing and experimentation code for every project.

## 4. Goals

### Primary goals

1. Reduce repetitive ML workflow implementation.
2. Make preprocessing decisions explicit and auditable.
3. Recommend treatments based on measurable dataset characteristics.
4. Prevent preprocessing leakage through reproducible pipelines.
5. Make model comparisons fair and reproducible.
6. Preserve human control over consequential decisions.
7. Produce artifacts that can be reused outside the platform.

### Secondary goals

- teach users why a treatment is recommended
- provide experiment history
- support future AI-assisted explanations
- provide a foundation for regression and more advanced AutoML

## 5. Non-goals

The product will not claim to:

- discover the objectively correct preprocessing strategy
- replace domain experts
- guarantee the best possible model
- automatically delete anomalous observations without approval
- use an LLM as the statistical decision engine
- solve every ML modality in the first release

## 6. Target users

### Persona A: ML learner

Needs:
- structured workflow
- explanations
- visibility into decisions
- generated code/pipeline

### Persona B: Data analyst moving into ML

Needs:
- reduced boilerplate
- guided preprocessing
- trustworthy model comparison

### Persona C: ML engineer

Needs:
- reproducibility
- experiment tracking
- exportable pipelines
- deterministic configuration

## 7. User journey

```text
Create Project
    ↓
Upload Dataset
    ↓
Select Target
    ↓
Automatic Profiling
    ↓
EDA Workspace
    ↓
Data Quality Findings
    ↓
Recommendations
    ↓
Human Review
    ↓
Pipeline Preview
    ↓
Train Experiments
    ↓
Compare Results
    ↓
Select Candidate
    ↓
Validate
    ↓
Export Model + Pipeline + Report
```

## 8. Functional requirements

### FR-001 Project management

The user can:
- create a project
- name a project
- describe a project
- upload datasets
- select a target column
- save workflow state

Acceptance criteria:
- project has a unique identifier
- project configuration is persisted
- dataset metadata is linked to the project

### FR-002 Dataset ingestion

Supported MVP:
- CSV

Requirements:
- detect delimiter
- infer encoding where practical
- report malformed rows
- record row/column counts
- preserve original dataset as immutable input

### FR-003 Dataset profiling

The system must detect:
- numerical columns
- categorical columns
- boolean columns
- datetime candidates
- identifier candidates
- constant/near-constant columns
- duplicate rows
- missing values
- cardinality
- basic distribution statistics

### FR-004 EDA

The system generates:
- missing-value summary
- class distribution
- numerical distributions
- boxplots
- categorical frequency charts
- correlation matrix
- target-vs-feature views where appropriate

EDA is primarily an observation layer. It must not silently modify data.

### FR-005 Missing-value recommendations

Candidate strategies:
- drop rows
- drop column
- mean imputation
- median imputation
- most-frequent imputation
- constant imputation
- KNN imputation, where technically appropriate

Recommendation inputs may include:
- missing percentage
- feature type
- skewness
- outlier rate
- cardinality
- sample size
- relationship with other features

The system must show the reason for each recommendation.

### FR-006 Outlier analysis

Detection methods:
- IQR
- Z-score
- robust statistics

Candidate treatments:
- leave unchanged
- row removal
- IQR-based filtering
- capping/winsorization
- robust scaling

The system must distinguish **outlier detection** from **outlier treatment**.

### FR-007 Encoding

Candidate methods:
- one-hot encoding
- ordinal encoding
- frequency encoding as a future extension
- target encoding only after leakage-safe implementation

High-cardinality features should generate a warning rather than an automatic destructive action.

### FR-008 Scaling

Candidate methods:
- StandardScaler
- MinMaxScaler
- RobustScaler
- no scaling

Recommendations should consider:
- model family
- distribution
- outlier presence
- feature magnitude

### FR-009 Data splitting

The system must split data before fitting preprocessing transformations.

Default:
- train/test split
- stratification for classification where appropriate

Cross-validation must fit transformations inside each training fold.

### FR-010 Model selection

Supported classifiers:
- Logistic Regression
- Linear SVM
- Random Forest Classifier
- Gradient Boosting Classifier
- HistGradientBoosting Classifier
- KNN Classifier
- XGBoost Classifier
- LightGBM Classifier

Supported regressors:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- KNN Regressor
- XGBoost Regressor
- LightGBM Regressor

### FR-011 Evaluation

Supported classification metrics:
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

Supported regression metrics:
- $R^2$ (coefficient of determination / explained variance)
- MAE (mean absolute error)
- MSE (mean squared error)
- RMSE (root mean squared error)
- residual analysis & actual-vs-predicted scatter plots

Evaluation protocol:
- StratifiedKFold cross-validation for classification
- KFold cross-validation for regression

The user selects a primary objective.

### FR-012 Experiment tracking

Every experiment must record:
- dataset version
- preprocessing configuration
- model configuration & hyperparameters
- random seed
- evaluation strategy (folds & split type)
- metrics (mean & std across folds)
- execution status & training time
- timestamp
- artifact references (pipeline.pkl, metrics.json)
- MLflow run tracking and model registration

### FR-013 Recommendation approval

Each recommendation has:
- status: proposed / accepted / rejected / overridden
- rationale
- evidence
- affected columns
- expected impact
- confidence
- task applicability (`NOT_APPLICABLE` for invalid task types)

### FR-014 Export

The user can export:
- trained model
- preprocessing pipeline
- complete Scikit-Learn Pipeline (`pipeline.pkl`)
- configuration JSON
- experiment results
- standalone reproducible Python training script (`train_pipeline.py`)
- standalone comprehensive HTML experiment report

### FR-015 Model Serving & Live Inference

The platform must provide low-latency model serving for candidate models:
- Dynamic input schema generation (`/models/candidate/schema`) matching active model features.
- Single-row live prediction and batch CSV upload evaluation.
- Classification confidence and class probability distribution.
- Regression uncertainty quantification: $R^2$ confidence score, margin of error ($\pm\text{MAE}$), and expected prediction intervals $[\text{pred} - \text{MAE}, \text{pred} + \text{MAE}]$.
- Automatic tolerance for dropped/unused features.

### FR-016 Model & Data Drift Monitoring

The platform continuously monitors production inferences against baseline training data:
- Asynchronous non-blocking logging of feature inputs and predictions (`PredictionLog`).
- Two-sample Kolmogorov-Smirnov Test and Wasserstein Distance for continuous numerical features.
- Population Stability Index (PSI) for categorical features.
- Aggregate system health: `HEALTHY`, `WARNING`, and `DRIFT_DETECTED`.
- Live traffic simulator for test validation.

### FR-017 Automated Feature Engineering & Hyperparameter Optimization

- Automated generation of candidate features: log1p skew transformations and pairwise numerical interactions.
- Systematic hyperparameter tuning using `RandomizedSearchCV` with interactive trial inspection and direct registration of tuned models.

## 9. Non-functional requirements

### Reproducibility

Given identical:
- dataset
- configuration
- library versions
- random seed

the system should produce equivalent results within expected numerical tolerance.

### Safety

No destructive data transformation occurs without explicit approval in interactive mode.

### Explainability

Recommendations must be backed by observable evidence.

### Performance

The MVP should handle ordinary educational/small-business tabular datasets without requiring distributed infrastructure.

### Extensibility

Detection, recommendation, transformation, and model components should be plugin-like modules.

## 10. UX requirements

Each workflow step should answer:

1. What did the system detect?
2. Why does it matter?
3. What options exist?
4. What does the system recommend?
5. What happens if I choose it?
6. Can I undo or change it?

## 11. Success metrics

Product metrics:
- time from upload to first valid experiment
- number of manual preprocessing steps avoided
- percentage of recommendations accepted
- percentage of recommendations overridden
- experiment reproducibility rate

Technical metrics:
- pipeline execution success rate
- preprocessing leakage incidents
- failed experiment rate
- recommendation consistency

## 12. Risks

### Risk: false confidence

A ranked recommendation may be interpreted as truth.

Mitigation:
- use language such as "recommended" rather than "correct"
- expose evidence
- expose alternatives

### Risk: data leakage

Mitigation:
- use sklearn Pipeline/ColumnTransformer
- fit transformations only on training folds

### Risk: destructive automation

Mitigation:
- preview changes
- explicit approval
- immutable raw dataset

### Risk: metric gaming

Mitigation:
- require primary metric selection before comparison
- show multiple metrics
- show train/test or CV behavior

## 13. MVP definition of done

The MVP is complete when a user can:

1. Upload a CSV.
2. Select a target.
3. Inspect automated profiling and EDA.
4. Review detected data-quality issues.
5. Choose or approve preprocessing strategies.
6. Generate a leakage-safe sklearn pipeline.
7. Train multiple baseline classifiers.
8. Compare them using cross-validation.
9. Select a candidate based on a defined objective.
10. Export the complete pipeline and experiment configuration.
