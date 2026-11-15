# Data Model

## Entities

### Tenant
- id
- name
- slug
- is_active
- created_at
- updated_at

### User
- id
- tenant_id (FK -> Tenant.id)
- email
- username
- hashed_password
- role (admin, data_scientist, viewer)
- is_active
- created_at

### ApiKey
- id
- tenant_id (FK -> Tenant.id)
- user_id (FK -> User.id)
- name
- prefix
- key_hash
- is_active
- expires_at
- created_at

### Project
- id
- tenant_id (FK -> Tenant.id)
- owner_id (FK -> User.id)
- name
- description
- task_type (classification, regression, sentiment_analysis)
- target_column
- status
- created_at
- updated_at
### Dataset
- id
- project_id
- filename
- storage_uri
- content_hash
- rows
- columns
- schema_json
- profile_json
- created_at

### WorkflowDecision
- id
- project_id
- recommendation_id
- category
- column
- selected_action
- previous_action
- status
- rationale
- created_at

### Experiment
- id
- project_id
- dataset_id
- config_json
- pipeline_hash
- model_name
- random_seed
- status
- started_at
- finished_at

### Metric
- id
- experiment_id
- name
- value
- split
- fold

### Artifact
- id
- experiment_id
- type
- storage_uri
- checksum


### Job (Distributed Asynchronous Queue)
- id
- project_id (FK -> Project.id)
- tenant_id (FK -> Tenant.id)
- task_name (experiment_runner, tuner, meta_learning_train, evaluate_candidates)
- status (pending, running, completed, failed, cancelled)
- priority
- payload_json
- result_json
- error_message
- progress (0.0 to 100.0)
- worker_id
- retry_count
- max_retries
- created_at
- started_at
- completed_at

### MetaRunRecord (Meta-Learning Historical Runs)
- id
- project_id (FK -> Project.id)
- dataset_id (FK -> Dataset.id)
- task_type
- meta_features_json
- strategy_name (recommended, baseline, alternative)
- primary_metric
- reference_score
- treatment_score
- lift
- validation_status
- created_at
### PipelineConfig
- id
- project_id
- config_json
- pipeline_hash
- created_at
- updated_at

### PredictionLog
- id
- project_id
- experiment_id
- features_json
- prediction_result
- created_at

## Relationships

```text
Project
 ├── Dataset
 ├── WorkflowDecision
 ├── PipelineConfig
 ├── PredictionLog
 └── Experiment
       ├── Metric
       └── Artifact
```

## Versioning

Dataset changes should create new dataset versions rather than mutate the original.

Pipeline configurations should also be immutable after experiment execution.
