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

### Class imbalance (Classification Only)

Signals:
- minority class proportion
- imbalance ratio
- metric selection
- *Note:* Strictly marked as `NOT_APPLICABLE` and omitted for regression tasks.

### Target Skewness & Outliers (Regression Only)

Signals:
- target skewness magnitude ($|\text{skew}| > 1.0$)
- minimum target value ($\ge 0$ for `log1p_transform`, general for `yeo_johnson_transform`)
- target outlier rate ($> 3\%$ with heavy tails)
- treatments: robust models (`HuberRegressor`, `Ridge`, `HistGradientBoostingRegressor`) and IQR target clipping

### Data Leakage (Priority: Critical, All Tasks)

Data leakage detection runs **before all other recommendations** and produces the highest-score recommendations (score ≥ 75, up to 95).

Two detection modes:

**1. Identifier Columns**

Signals:
- `dtype == "identifier"` from profiler, OR
- `unique_ratio >= 0.95` for integer / string columns (excluding float columns)
- name contains keywords: `id`, `uuid`, `guid`, `pk`, `key`, `code`, `hash`, `token`, `index`, `row`

Severity: always `critical` (score = 95, confidence = 0.97)

Recommended action: `drop_column` before training.

**2. Target Proxy Columns**

Signals (numeric feature vs numeric/binary target):
- Pearson correlation $|\rho| \ge 0.95$
- Spearman correlation $|\rho_s| \ge 0.95$
- Point-Biserial $|r_{pb}| \ge 0.95$ (when target is binary)

Signals (categorical feature vs categorical target):
- Cramér's V $\ge 0.90$

Severity mapping:
- `critical` if correlation ≥ 0.98 (score = 95)
- `high` if correlation ≥ 0.95 (score = 88)
- `medium` if correlation ≥ threshold (score = 75)

Recommended action: `drop_column` with `verify_manually` as alternative.

**Exclusion rule for identifier detection:**
Float columns are never classified as identifiers even with `unique_ratio = 1.0`,
because continuous measurements (income, price, temperature) naturally produce
unique float values without being IDs.



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
