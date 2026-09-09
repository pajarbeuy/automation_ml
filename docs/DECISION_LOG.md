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
