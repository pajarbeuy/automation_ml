# Testing Strategy

## 1. Unit tests

Test:
- type detection
- missing-value statistics
- skewness calculation
- outlier detection
- recommendation scoring
- pipeline generation
- metric calculation

## 2. Integration tests

Test complete flows:

```text
CSV
→ profile
→ recommendations
→ decisions
→ pipeline
→ experiment
→ artifact
```

## 3. Leakage tests

Create synthetic datasets where leakage is easy to detect.

Verify that:
- imputation is fit per training fold
- scaling is fit per training fold
- encoders are fit per training fold
- feature selection is fit per training fold

## 4. Reproducibility tests

Same:
- dataset
- configuration
- seed
- environment

should produce equivalent results.

## 5. Recommendation tests

For known synthetic scenarios, assert reasonable recommendations.

Example:
- strongly skewed numeric feature + outliers → median should rank above mean

Do not assert that one treatment is universally correct.

## 6. UI tests

Verify:
- upload
- target selection
- recommendation approval
- pipeline preview
- experiment comparison
- artifact export

## 7. Regression tests

Every discovered bug becomes a permanent test where practical.
