# Data Model

## Entities

### Project
- id
- name
- description
- task_type
- target_column
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

## Relationships

```text
Project
 ├── Dataset
 ├── WorkflowDecision
 └── Experiment
       ├── Metric
       └── Artifact
```

## Versioning

Dataset changes should create new dataset versions rather than mutate the original.

Pipeline configurations should also be immutable after experiment execution.
