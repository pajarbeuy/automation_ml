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

## Artifacts

### GET /api/experiments/{experiment_id}/artifacts
List artifacts.

### GET /api/artifacts/{artifact_id}
Download/stream artifact through authorized access.

## API design rules

- JSON responses
- Pydantic validation
- consistent error schema
- pagination for collections
- project-scoped authorization
- idempotency for expensive job submission where needed
