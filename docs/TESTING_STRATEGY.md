# Testing Strategy

## 1. Testing Objectives

Testing is performed to ensure that the AutoML platform is:

* Functionally correct
* Task-aware
* Evidence-driven
* Model-aware
* Leakage-safe
* Reproducible
* Robust against invalid and unusual datasets
* Consistent across API, ML engine, database, and UI
* Explainable and auditable
* Safe for production workflows

Testing must validate not only whether a component executes successfully, but also whether it produces the correct decision and expected behavior.

The primary testing principle is:

> A successful execution does not necessarily mean a correct ML decision.

---

# 2. Testing Levels

The platform uses multiple testing levels.

```text
Unit Tests
    ↓
Integration Tests
    ↓
API / Contract Tests
    ↓
ML Decision Tests
    ↓
End-to-End Tests
```

Each level has a different purpose and must not replace the others.

---

# 3. Test Case Classification

Test cases are classified using the following categories.

## 3.1 Positive Tests

Verify that valid inputs produce the expected successful behavior.

Examples:

* Valid CSV is accepted
* Valid regression dataset is detected as regression
* Valid missing-value scenario produces applicable recommendations
* Valid pipeline is generated successfully
* Valid experiment produces evaluation metrics

---

## 3.2 Negative Tests

Verify that invalid or unsupported conditions are handled correctly.

Examples:

* Invalid file format
* Missing target column
* Unsupported target type
* Invalid task configuration
* Invalid pipeline configuration
* Invalid prediction payload
* Regression dataset receiving classification-only treatment

Expected behavior must include:

* Correct error or state
* No invalid downstream operation
* No corrupted persistence state

---

## 3.3 Boundary Tests

Verify behavior at or around configured thresholds.

Examples:

```text
missing_ratio = threshold - ε
missing_ratio = threshold
missing_ratio = threshold + ε
```

Other boundary cases include:

* Minimum dataset size
* Maximum file size
* Minimum number of classes
* Maximum cardinality
* Zero variance
* Outlier threshold
* HPO search-space limits

---

## 3.4 Edge Case Tests

Verify valid but unusual or extreme inputs.

Examples:

* Empty dataset
* Single-row dataset
* Very small dataset
* Constant target
* Constant feature
* All values missing
* Single categorical value
* Extremely high-cardinality feature
* Extreme numerical values
* Unseen categorical values
* Missing prediction features

---

## 3.5 Error Handling Tests

Verify that failures are handled safely and consistently.

Test:

* Correct HTTP status codes
* Error response schema
* Validation messages
* Internal exception handling
* Logging
* No partial or corrupted artifacts

---

# 4. Unit Testing

Unit tests verify isolated components independently from external services.

## 4.1 Profiling

Test:

* Type detection
* Numerical/categorical/boolean/datetime detection
* Missing-value statistics
* Duplicate detection
* Cardinality
* Constant features
* Near-constant features
* Skewness
* Kurtosis
* Quantiles
* Variance
* IQR
* Outlier detection
* Target statistics

Test categories:

* Positive
* Negative
* Boundary
* Edge case

---

## 4.2 Task Detection

Test:

* Classification detection
* Regression detection
* Time-series detection
* Ambiguous datasets
* Low-cardinality numerical targets
* High-cardinality categorical targets
* Datetime presence
* User task override

The system must not incorrectly classify a regression problem as classification solely because the target has a small number of unique values.

---

## 4.3 Recommendation Engine

Test:

* Recommendation rule evaluation
* Rule applicability
* Recommendation scoring
* Confidence calculation
* Evidence generation
* Severity
* Alternative treatments
* Recommendation state
* Task-specific recommendations

Particular attention must be given to:

```text
Classification
    ≠
Regression
    ≠
Time Series
```

Classification-only recommendations must not become actionable recommendations for regression.

---

## 4.4 Pipeline Generation

Test:

* ColumnTransformer generation
* Numeric pipeline
* Categorical pipeline
* Imputation
* Encoding
* Scaling
* Outlier treatment
* Pipeline serialization
* Pipeline deserialization
* Pipeline configuration validation

---

## 4.5 Experiment Engine

Test:

* Train/test splitting
* Stratified splitting
* KFold
* Model initialization
* Cross-validation
* Hyperparameter search
* Random seed handling
* Experiment persistence
* Metric calculation
* Result ranking

---

## 4.6 Evaluation

Classification:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Confusion matrix

Regression:

* R²
* MAE
* MSE
* RMSE
* MAPE when applicable
* sMAPE when applicable
* Residual analysis
* Actual vs predicted analysis

Metric tests must use deterministic synthetic data with known expected results.

---

# 5. API Contract Testing

API tests verify that backend endpoints follow their defined request and response contracts.

Test:

* Required fields
* Optional fields
* Data types
* Enum values
* Missing fields
* Invalid fields
* Response schema
* HTTP status codes
* Error response schema
* Authentication/authorization where applicable
* Invalid resource IDs

Example:

```text
POST /projects/{id}/predict

Valid payload
    → 200

Missing required feature
    → 4xx

Invalid feature type
    → 4xx

Unknown project
    → 404

Invalid model
    → appropriate error
```

API contract tests must ensure that backend changes do not silently break frontend consumers.

---

# 6. Integration Testing

Integration tests verify interactions between multiple platform components.

Primary workflow:

```text
CSV Upload
    ↓
Dataset Storage
    ↓
Profiling
    ↓
Task Detection
    ↓
Recommendations
    ↓
User Decisions
    ↓
Pipeline Generation
    ↓
Experiment
    ↓
Evaluation
    ↓
Artifact Persistence
```

Test:

* Data passed correctly between components
* IDs remain consistent
* Configuration is preserved
* Profile data is available to recommendation engine
* Recommendations are correctly converted into pipeline configuration
* Pipeline is correctly passed to experiment engine
* Experiment results are correctly persisted
* Artifacts can be retrieved

---

# 7. ML Decision Correctness Testing

This is a critical testing layer for the platform.

The test must verify not only that a recommendation is generated, but that the recommendation is logically appropriate given the available evidence.

Example:

```text
Dataset:
Regression

Evidence:
outlier_ratio = 15%

Expected:
RobustScaler may be generated as a candidate
```

Another example:

```text
Dataset:
Regression

Expected:
Class imbalance treatment
→ NOT_APPLICABLE
```

Tests must verify:

* Correct task
* Correct evidence
* Correct applicability
* Correct recommendation
* Correct severity
* Correct confidence
* Correct alternatives
* Correct validation requirement

The system must not recommend a treatment merely because a statistical condition exists.

---

# 8. Recommendation Testing

Recommendation tests use deterministic synthetic datasets representing known scenarios.

Examples:

## Missing Values

```text
Scenario:
Numerical feature contains moderate missing values

Expected:
Mean/median imputation candidates may be generated
```

## Outliers

```text
Scenario:
Feature contains significant extreme values

Expected:
Outlier warning
Robust treatment candidate
```

## Scaling

```text
Scenario:
KNN + features with significantly different scales

Expected:
Scaling candidate
```

```text
Scenario:
Tree-based model

Expected:
Scaling should not automatically be treated as required
```

## Classification Imbalance

```text
Scenario:
Severely imbalanced classification target

Expected:
class_weight / sampling candidates
```

## Regression Target

```text
Scenario:
Strongly skewed regression target

Expected:
target transformation candidates
```

Tests must verify that recommendations are **reasonable given the evidence**, not that a specific treatment is universally optimal.

---

# 9. Leakage Testing

Leakage testing is mandatory because leakage can produce artificially strong model performance.

## 9.1 Preprocessing Leakage

Verify that the following are fitted only using training data:

* Imputation
* Scaling
* Encoding
* Feature selection
* Feature engineering
* Target transformation
* Oversampling

The test must ensure that validation/test statistics cannot influence preprocessing.

---

## 9.2 Feature / Target Leakage

Create synthetic datasets containing obvious leakage.

Example:

```text
target = price

feature = price_after_tax
```

Expected:

```text
Potential leakage detected
```

Also test false-positive scenarios.

Example:

```text
house_area → price
```

A strong correlation alone must not automatically be classified as leakage.

---

## 9.3 Cross-Validation Leakage

Verify that transformations are fitted independently inside each training fold.

Incorrect:

```text
Fit transformer on entire dataset
        ↓
Cross-validation
```

Expected architecture:

```text
Fold 1
  Train → fit transformer
  Validation → transform

Fold 2
  Train → fit transformer
  Validation → transform
```

---

# 10. Reproducibility Testing

The same experiment configuration should produce equivalent results.

Given:

```text
Same dataset
Same configuration
Same model
Same preprocessing
Same CV strategy
Same random seed
Same environment
```

Expected:

```text
Equivalent splits
Equivalent metrics
Equivalent model configuration
Equivalent experiment metadata
```

Floating-point comparisons should use appropriate numerical tolerance where necessary.

---

# 11. Data Robustness Testing

Test datasets that are valid but difficult to process.

Examples:

* Empty target
* Constant target
* Constant features
* Extremely small datasets
* Missing target values
* All-missing columns
* High-cardinality categories
* Unseen categories
* Mixed types
* Invalid numerical values
* Extremely large numerical values
* Duplicate rows
* Duplicate timestamps
* Missing periods

The system must fail safely when a dataset cannot support a particular operation.

---

# 12. Serving Tests

Prediction serving must be tested independently and end-to-end.

Test:

* Single prediction
* Batch prediction
* Valid payload
* Missing features
* Extra features
* Invalid feature types
* Unseen categories
* Empty payload
* Invalid model
* Invalid project
* Regression output
* Classification output
* Confidence/uncertainty output
* Prediction interval output where applicable

---

# 13. Drift Monitoring Tests

Test:

* KS-Test
* Wasserstein distance
* PSI
* Drift threshold
* No-drift scenario
* Significant-drift scenario
* Aggregation
* Background logging
* Drift simulator
* Dashboard data

Boundary tests should verify behavior immediately below, at, and above configured drift thresholds.

---

# 14. UI / E2E Testing

The UI must be tested based on real user workflows.

Primary workflow:

```text
Upload Dataset
    ↓
Select Target
    ↓
Review Profile
    ↓
Review Recommendations
    ↓
Accept / Reject / Override
    ↓
Review Pipeline
    ↓
Run Experiment
    ↓
Compare Models
    ↓
Select Result
    ↓
Export Artifact
    ↓
Make Prediction
```

Test:

* Upload
* Target selection
* Profile visualization
* Recommendation display
* Evidence display
* Accept
* Reject
* Override
* NOT_APPLICABLE state
* Pipeline preview
* Experiment execution
* Experiment comparison
* Artifact export
* Prediction playground
* Error states
* Loading states

---

# 15. Regression Testing

Every confirmed bug should become a regression test when practical.

Workflow:

```text
Bug discovered
      ↓
Root cause identified
      ↓
Bug fixed
      ↓
Regression test created
      ↓
Test becomes permanent
```

Regression tests must prevent previously fixed issues from reappearing.

---

# 16. Performance Testing

Performance testing should be introduced progressively.

Test:

* Dataset profiling time
* Memory consumption
* API response time
* Pipeline construction time
* Experiment execution time
* Hyperparameter optimization time
* Prediction latency
* Batch prediction throughput
* Drift monitoring overhead

Performance thresholds should be defined according to dataset size and workload.

Example:

```text
Small dataset
    → profiling should complete within expected interactive latency

Large dataset
    → long-running operations should use asynchronous processing
```

---

# 17. Security Testing

Test:

* File upload validation
* File size limits
* Malformed files
* Invalid filenames
* Path traversal
* Unauthorized resource access
* Invalid project IDs
* Invalid model IDs
* Request validation
* API authentication once implemented
* Authorization once implemented

Uploaded files must never allow access outside the configured storage directory.

---

# 18. Test Data Strategy

Testing should use deterministic datasets designed for specific scenarios.

Test datasets should include:

```text
classification_balanced.csv
classification_imbalanced.csv
classification_high_cardinality.csv

regression_normal.csv
regression_skewed.csv
regression_outliers.csv
regression_constant_target.csv

missing_values.csv
duplicate_rows.csv
constant_features.csv
leakage_dataset.csv

small_dataset.csv
invalid_dataset.csv
edge_case_dataset.csv
```

Synthetic datasets should be preferred when the expected behavior can be explicitly controlled.

Real-world datasets may be used for integration and exploratory validation.

---

# 19. Requirement Traceability

Every important requirement should map to one or more test cases.

```text
Requirement
    ↓
Test Case
    ↓
Implementation
    ↓
Test Execution
    ↓
Result
```

Example:

```text
REQ-REC-001
Recommendation must be task-aware

        ↓

TC-REC-001
Regression must not receive class imbalance treatment

TC-REC-002
Classification must support imbalance recommendations

TC-REC-003
Time-series must use temporal-specific recommendations
```

This ensures that passing tests represent actual product requirements rather than arbitrary code coverage.

---

# 20. Test Priority

Test cases should be prioritized based on risk.

## Critical

* Data leakage
* Incorrect task detection
* Incorrect recommendation applicability
* Incorrect model evaluation
* Incorrect model selection
* Data corruption
* Unauthorized access

## High

* Pipeline generation
* Experiment execution
* Prediction serving
* Artifact persistence
* Recommendation validation

## Medium

* Visualization
* UI interactions
* Report generation
* Secondary metrics

## Low

* Cosmetic UI behavior
* Non-critical presentation details

---

# 21. Test Automation Strategy

Tests should be automated whenever behavior is deterministic and repeatable.

```text
Unit
    → pytest

API
    → pytest / HTTP client

Integration
    → pytest

ML correctness
    → pytest

E2E UI
    → Playwright

Frontend
    → Vitest / TypeScript compiler

```

Tests requiring external network services, GPU, or third-party APIs should be isolated or mocked where practical.

---

# 22. Quality Gates

A feature should not be considered complete merely because its implementation exists.

Minimum quality gate:

```text
Implementation
    +
Unit Tests
    +
Negative Tests
    +
Boundary / Edge Tests
    +
Integration Test
    +
Regression Test where applicable
```

Critical ML features additionally require:

```text
ML Decision Correctness
    +
Leakage Test
    +
Reproducibility Test
```

A recommendation or model-selection feature should not be considered production-ready if its decision logic cannot be validated.

---

# 23. Active Test Suite Inventory

The project maintains an automated test inventory covering the major platform layers.

Current test modules include:

* `test_profiling.py`
* `test_recommendations.py`
* `test_leakage_detector.py`
* `test_pipeline.py`
* `test_experiments.py`
* `test_evaluator_regression.py`
* `test_feature_engineering.py`
* `test_tuner.py`
* `test_export.py`
* `test_serving.py`
* `test_drift_detector.py`
* `test_robustness_serving.py`
* `test_robustness_datasets.py`
* `test_scaling_preview.py`
* `test_assistant.py`
* `test_api_contracts.py`
* `test_registry.py`

The test count should be treated as an inventory metric rather than the primary quality metric.

Quality should instead be evaluated through:

```text
Requirement Coverage
+
Risk Coverage
+
ML Decision Correctness
+
Integration Coverage
+
Regression Coverage
```

---

# 24. Definition of Done for Testing

A feature is considered adequately tested when:

* Positive scenarios pass
* Negative scenarios pass
* Relevant boundary cases pass
* Relevant edge cases pass
* Error handling behaves correctly
* Unit tests pass
* Integration tests pass where applicable
* ML decision correctness is verified where applicable
* Leakage risks are tested where applicable
* Reproducibility is verified where applicable
* Regression tests exist for important historical bugs
* Requirement traceability is established
* No critical test failures remain

The goal of testing is not to maximize the number of tests.

The goal is to provide sufficient evidence that the system behaves correctly under expected, invalid, boundary, unusual, and adversarial conditions.

---

# 25. Testing Strategy for Autonomous Sentiment Analysis

Sentiment analysis introduces distinct risks related to vocabulary leakage, token parsing, negation destruction, out-of-vocabulary (OOV) shifts, and high-dimensional sparse linear classification. The following testing protocols are mandatory:

## 25.1 Zero-Leakage NLP Cross-Validation Protocol
* **Core Requirement**: The vocabulary vocabulary dictionary and inverse document frequency (IDF) weights must be learned **strictly from the training fold**.
* **Verification**: In unit tests, verify that tokens present exclusively in validation folds never appear in `vectorizer.vocabulary_`.
* **Failure Criteria**: Any leakage of validation n-grams into training vocabulary fails the test suite immediately.

## 25.2 Negation & Semantic Preservation Tests
* **Core Requirement**: Stopword pruning or token normalization must not eliminate critical negation qualifiers (e.g. `not`, `never`, `tidak`, `bukan`).
* **Verification**:
  * Sentence A: *"Produk ini sangat bagus dan berfungsi normal."* $\rightarrow$ Predicted class: **Positive**.
  * Sentence B: *"Produk ini **tidak** bagus dan **tidak** berfungsi normal."* $\rightarrow$ Predicted class: **Negative**.
  * Test asserts that Sentence B does not map to Positive sentiment.

## 25.3 Adversarial, Extreme, and Malformed Text Tests
* **Empty & Whitespace Inputs**: Inputs such as `""`, `"   "`, `"\t\n"`, or missing/null text must be gracefully handled (either imputed with `"[EMPTY]"` or rejected with HTTP 422, never crashing with HTTP 500).
* **Emoji & Punctuation Only**: Inputs consisting purely of punctuation or emojis (e.g. `":)", "???", "🔥👍"`) must be processed deterministically.
* **Length Boundaries**: Teks very short (single character / 1 token) and very long (10,000+ words) must execute within memory limits without stack overflow or timeout.

## 25.4 Out-Of-Vocabulary (OOV) Resilience Tests
* **Core Requirement**: Inference payloads containing 100% unseen/alien words (e.g. random alphanumeric hashes `x9f8k21` or foreign language strings) must yield a valid probability distribution or neutral class rather than raising an index error.

## 25.5 Word Attribution & Serving API Contract Tests
* **Core Requirement**: The prediction endpoint `/api/projects/{id}/predict` for sentiment tasks must return a validated schema containing:
  * `prediction`: predicted sentiment label (`positive`, `negative`, `neutral`).
  * `confidence`: float in range $[0.0, 1.0]$.
  * `top_contributing_words`: array of objects `{ "word": str, "weight": float, "sentiment": str }`.
* **Verification**: Ensure the sum of top positive vs negative weights matches the directional orientation of the final decision.
