// Types matching the backend Pydantic schemas

export interface Project {
  id: string;
  name: string;
  description: string | null;
  task_type: string;
  target_column: string | null;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  filename: string;
  content_hash: string | null;
  rows: number | null;
  columns: number | null;
  created_at: string;
}

export interface FeatureProfile {
  name: string;
  dtype: string;
  missing_count: number;
  missing_rate: number;
  unique_count: number;
  constant: boolean;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  p25?: number | null;
  p50?: number | null;
  p75?: number | null;
  skewness?: number | null;
  kurtosis?: number | null;
  outlier_rate_iqr?: number | null;
  outlier_rate_zscore?: number | null;
  top_categories: Array<{ value: string; frequency: number }>;
  high_cardinality: boolean;
}

export interface DatasetProfile {
  id: string;
  filename: string;
  rows: number | null;
  columns: number | null;
  profile_json: {
    rows: number;
    columns: number;
    target: string;
    duplicate_rows: number;
    features: FeatureProfile[];
    class_distribution: Record<string, number>;
    dataset_hash: string;
  } | null;
}

export interface EDAData {
  missing_value_summary: Array<{ column: string; missing_count: number; missing_rate: number }>;
  class_distribution: Array<{ class: string; proportion: number }>;
  numerical_distributions: Record<string, {
    bins: number[];
    counts: number[];
    mean: number | null;
    median: number | null;
    std: number | null;
    skewness: number | null;
    outlier_rate_iqr: number;
  }>;
  boxplots: Record<string, {
    min: number | null;
    q1: number | null;
    median: number | null;
    q3: number | null;
    max: number | null;
    lower_fence: number | null;
    upper_fence: number | null;
    outliers: number[];
  }>;
  categorical_frequencies: Record<string, Array<{ value: string; frequency: number }>>;
  correlation_matrix: { columns: string[]; matrix: (number | null)[][] };
  target_vs_features: Record<string, {
    type: string;
    data: Record<string, number | null> | Array<Record<string, string | number>>;
  }>;
}

export interface DatasetEDA {
  id: string;
  filename: string;
  eda_json: EDAData | null;
}

export interface Recommendation {
  id: string;
  category: string;
  column: string | null;
  recommended_action: string;
  confidence: number;
  score: number;
  evidence: string[];
  rationale: string;
  alternatives: string[];
  status: string;
}

export interface WorkflowDecision {
  id: string;
  recommendation_id: string | null;
  category: string;
  column_name: string | null;
  selected_action: string;
  status: string;
  rationale: string | null;
  created_at: string;
}

export interface PipelineConfig {
  id: string;
  project_id: string;
  config_json: {
    numeric_columns: string[];
    categorical_columns: string[];
    dropped_columns: string[];
    numeric_config: Record<string, { missing: string; outlier: string; scaling: string }>;
    categorical_config: Record<string, { missing: string; encoding: string }>;
    model: { name: string; params: Record<string, unknown> };
  };
  pipeline_hash: string | null;
  updated_at: string;
}

export interface PipelineValidation {
  valid: boolean;
  steps: string[];
  warnings: string[];
  errors: string[];
}

export interface Metric {
  id: string;
  name: string;
  value: number;
  split: string | null;
  fold: number | null;
}

export interface Artifact {
  id: string;
  type: string;
  storage_uri: string;
  checksum: string | null;
  created_at: string;
}

export interface Experiment {
  id: string;
  project_id: string;
  dataset_id: string;
  model_name: string;
  random_seed: number;
  status: string;
  training_time_seconds: number | null;
  error_message: string | null;
  config_json: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  metrics: Metric[];
  artifacts: Artifact[];
}

export interface ExperimentCreate {
  model_names: string[];
  random_seed?: number;
  cv_folds?: number;
  primary_metric?: string;
  test_size?: number;
  enable_feature_engineering?: boolean;
  is_tuned?: boolean;
}

export interface ScalingStats {
  bins: number[];
  counts: number[];
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  skewness: number | null;
}

export interface ScalingColumnPreview {
  column_name: string;
  scaling_action: string;
  raw: ScalingStats;
  transformed: ScalingStats;
}

export type ScalingPreviewData = Record<string, ScalingColumnPreview>;

export interface TuningTrial {
  trial_index: number;
  params: Record<string, string>;
  score: number;
  duration_seconds: number;
}

export interface TuningResultData {
  model_name: string;
  best_params: Record<string, unknown>;
  best_score: number;
  primary_metric: string;
  trials: TuningTrial[];
  total_time_seconds: number;
}

