# Recommendation Engine Specification

## 1. Philosophy

The recommendation engine is not a collection of arbitrary if/else rules.

It is a constrained decision system that maps:

```text
dataset evidence
+ task context
+ model context
+ user constraints
        ↓
candidate actions
        ↓
ranking
        ↓
explanation
```

## 2. Recommendation lifecycle

```text
Detect
  ↓
Generate candidates
  ↓
Estimate suitability
  ↓
Rank
  ↓
Explain
  ↓
Human review
  ↓
Approve / reject / override
```

## 3. Evidence categories

### Missing values

Signals:
- missing rate
- numeric/categorical type
- skewness
- outlier rate
- sample size
- feature correlation

### Outliers

Signals:
- IQR rate
- robust z-score
- distribution shape
- sample size
- model sensitivity

### Encoding

Signals:
- categorical cardinality
- ordinal semantics
- number of categories
- training size

### Scaling

Signals:
- model family
- feature ranges
- outliers
- distribution

### Class imbalance

Signals:
- minority class proportion
- imbalance ratio
- metric selection

## 4. Recommendation scoring

Do not pretend the score is a probability unless calibrated.

Example:

```text
score =
  evidence_fit
+ robustness
+ simplicity
+ model_compatibility
- information_loss_risk
```

The output can be normalized to 0-100 for UI presentation, but the UI should call it a **recommendation score**, not certainty.

## 5. Example: missing values

For numeric features:

Potential recommendation logic:

```text
if missing_rate == 0:
    no_action

elif missing_rate is very high:
    warn about dropping feature

elif skewness is high or outlier_rate is high:
    median gets preference

else:
    mean or median may both be reasonable
```

Important:
Thresholds must be configurable and validated empirically.

## 6. Example: outliers

The engine should never infer:

```text
outlier == error
```

Instead:

```text
outlier detected
      ↓
potential treatment
      ↓
human/domain decision
```

## 7. Recommendation explanation

Every recommendation must expose:

- detected condition
- relevant statistics
- recommended option
- alternatives
- trade-offs
- confidence/score
- affected columns

## 8. Recommendation evaluation

The recommendation engine itself must be testable.

Evaluation dimensions:
- consistency
- safety
- usefulness
- acceptance rate
- override rate
- downstream performance impact

A recommendation that improves CV score but causes information loss should not automatically be considered successful.

## 9. Future: learned recommendation engine

Once enough experiments exist, recommendations could be learned from historical experiments.

```text
Dataset characteristics
       ↓
Recommendation model
       ↓
Expected downstream performance
```

This is a later research direction, not an MVP requirement.
