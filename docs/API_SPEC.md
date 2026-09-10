# API Specification

## Projects

### POST /api/projects
Create a project.

### GET /api/projects
List projects.

### GET /api/projects/{project_id}
Get project details.

### DELETE /api/projects/{project_id}
Delete project.

## Datasets

### POST /api/projects/{project_id}/datasets
Upload dataset.

### GET /api/datasets/{dataset_id}/profile
Return dataset profile.

### GET /api/datasets/{dataset_id}/eda
Return EDA metadata and visualization references.

## Recommendations

### GET /api/datasets/{dataset_id}/recommendations
Return recommendations.

### POST /api/recommendations/{recommendation_id}/decision
Accept/reject/override a recommendation.

Request:

```json
{
  "action": "median_imputation",
  "reason": "Distribution is skewed and outliers are present."
}
```

## Pipelines

### POST /api/projects/{project_id}/pipeline
Generate pipeline from approved decisions.

### GET /api/projects/{project_id}/pipeline
Return current pipeline configuration.

### POST /api/projects/{project_id}/pipeline/validate
Validate pipeline without training.

## Experiments

### POST /api/projects/{project_id}/experiments
Run experiment.

### GET /api/projects/{project_id}/experiments
List experiments.

### GET /api/experiments/{experiment_id}
Return experiment details.

### GET /api/experiments/{experiment_id}/metrics
Return metrics.

### POST /api/projects/{project_id}/experiments/tune
Run randomized hyperparameter tuning search for candidate models.

## Artifacts

### GET /api/experiments/{experiment_id}/artifacts
List artifacts.

### GET /api/artifacts/{artifact_id}
Download/stream artifact through authorized access.

### GET /api/experiments/{experiment_id}/export/python
Download standalone reproducible Python training code.

### GET /api/experiments/{experiment_id}/export/report
Download standalone comprehensive HTML experiment report.

## Model Serving & Live Inference

### GET /api/projects/{project_id}/models/candidate/schema
Return expected input feature schema (names, dtypes, categories, statistics) excluding dropped and target columns.

### POST /api/projects/{project_id}/predict
Execute live model prediction on raw JSON features.
- Accepts single object `features` or array of objects `instances`.
- Logs features and predictions asynchronously in the background for drift monitoring.
- For classification: returns `prediction`, `confidence`, and `probabilities`.
- For regression: returns `prediction`, `confidence` ($R^2$ Score), `margin_of_error` ($\pm\text{MAE}$), `interval_lower`, and `interval_upper`.

### POST /api/projects/{project_id}/predict/batch-csv
Upload CSV containing feature columns, run inference on all rows, and return enriched CSV with `prediction`, `confidence`, and regression interval columns.

## Model & Data Drift Monitoring

### GET /api/projects/{project_id}/drift
Compute and return real-time feature drift analysis between training baseline and production inference logs using KS-Test, Wasserstein Distance, and PSI.

### POST /api/projects/{project_id}/drift/simulate
Simulate production traffic (`normal` or `drifted`) for demonstration and testing.

### DELETE /api/projects/{project_id}/drift/logs
Reset inference production logs for the specified project.

## Preview & Explainability

### GET /api/projects/{project_id}/scaling-preview
Return distribution histogram comparison (Original vs Scaled) for numerical columns.

### POST /api/assistant/explain-recommendation
Return rule-based plain English and Indonesian explanation for a preprocessing recommendation.

### GET /api/assistant/guidance
Return context-aware next step guidance for the user's current project state.

## API design rules

- JSON responses
- Pydantic validation
- consistent error schema
- non-blocking asynchronous logging for inference monitoring
- pagination for collections
- project-scoped authorization
- idempotency for expensive job submission where needed
