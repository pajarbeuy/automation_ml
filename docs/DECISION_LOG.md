# Decision Log

This file records architectural and product decisions.

## ADR-001: Human-in-the-loop

**Decision:** Consequential preprocessing decisions require explicit user control.

**Reason:** Statistical detection cannot determine domain semantics.

## ADR-002: sklearn Pipeline

**Decision:** Use sklearn Pipeline and ColumnTransformer as the core execution abstraction.

**Reason:** Reduces leakage risk and provides reproducible transformations.

## ADR-003: Deterministic recommendation engine

**Decision:** Recommendations originate from deterministic/statistical evidence.

**Reason:** LLM output is not sufficiently reliable as a numerical decision engine.

## ADR-004: Immutable raw dataset

**Decision:** Original uploaded data is never modified.

**Reason:** Enables reproducibility, rollback, and auditability.

## ADR-005: Primary metric selected before experiments

**Decision:** User defines the optimization objective before comparing models.

**Reason:** Prevents post-hoc metric selection.

## ADR-006: MVP starts with classification

**Decision:** Classification is the first supported task.

**Reason:** It provides a constrained but rich test bed for preprocessing, imbalance, metrics, and model comparison.

## ADR-007: Task-aware preprocessing recommendation engine

**Decision:** The recommendation engine strictly differentiates between `classification` and `regression` tasks.

**Reason:** Concepts such as class imbalance, minority classes, and SMOTE are mathematically inapplicable to continuous numerical targets. For regression targets, the engine replaces class imbalance with continuous target skewness detection (log1p/Yeo-Johnson transformation) and target outlier detection (robust regressors / IQR clipping).

## ADR-008: Confidence and uncertainty quantification for regression serving

**Decision:** In addition to point estimates, regression model serving provides $R^2$ explained variance as model confidence, $\pm\text{MAE}$ as margin of error, and estimated prediction intervals $[\text{pred} - \text{MAE}, \text{pred} + \text{MAE}]$.

**Reason:** Unlike classification models that output softmax class probabilities, continuous regression models output single scalar point predictions. Without explicit error margins and variance-explained indicators, users may incorrectly assume the model output is 100% exact and infallible. Providing confidence metrics and prediction intervals communicates statistical uncertainty clearly.
