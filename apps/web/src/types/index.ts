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

export interface TextProfile {
  text_column: string;
  doc_count: number;
  empty_docs_count: number;
  total_tokens: number;
  vocab_size: number;
  lexical_richness: number;
  stopword_ratio: number;
  uppercase_ratio: number;
  punctuation_ratio: number;
  char_stats?: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std: number;
  };
  token_stats?: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std: number;
  };
  top_unigrams?: Array<{ token: string; count: number; frequency: number }>;
  top_bigrams?: Array<{ bigram: string; count: number }>;
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
    text_profile?: TextProfile | null;
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
  experiment_name?: string | null;
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
  experiment_name?: string;
  random_seed?: number;
  cv_folds?: number;
  primary_metric?: string;
  test_size?: number;
  enable_feature_engineering?: boolean;
  is_tuned?: boolean;
  model_params?: Record<string, any>;
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

// ─── Model Serving & Prediction Types ─────────────────────────
export interface FeatureSchema {
  name: string;
  dtype: 'numeric' | 'categorical';
  example_value: any;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  categories?: string[];
}

export interface CandidateSchemaResponse {
  project_id: string;
  candidate_experiment_id: string;
  model_name: string;
  task_type: string;
  target_column: string;
  features: FeatureSchema[];
}

export interface WordAttributionItem {
  word: string;
  weight: number;
  sentiment: 'positive' | 'negative' | string;
}

export interface PredictionResult {
  prediction: string | number;
  confidence?: number;
  confidence_metric?: string;
  probabilities?: Record<string, number>;
  margin_of_error?: number;
  interval_lower?: number;
  interval_upper?: number;
  contributing_words?: WordAttributionItem[];
}
export interface PredictResponse {
  result?: PredictionResult;
  results?: PredictionResult[];
  count?: number;
  model_name: string;
  task_type: string;
  target_column: string;
}

// ─── Model Drift Monitoring Types ─────────────────────────────
export interface DriftDistributionPoint {
  bin: string;
  baseline_pct: number;
  current_pct: number;
}

export interface FeatureDriftDetail {
  feature_name: string;
  dtype: 'numeric' | 'categorical';
  test_name: string;
  statistic: number;
  p_value: number | null;
  threshold: number;
  wasserstein_distance?: number;
  baseline_mean?: number;
  current_mean?: number;
  baseline_std?: number;
  current_std?: number;
  drift_detected: boolean;
  status: 'STABLE' | 'MODERATE' | 'DRIFT' | 'INSUFFICIENT_DATA';
  distribution_chart: DriftDistributionPoint[];
}

export interface DriftReportResponse {
  project_id?: string;
  status: 'HEALTHY' | 'WARNING' | 'DRIFT_DETECTED' | 'INSUFFICIENT_DATA';
  total_predictions_logged: number;
  min_required?: number;
  monitored_features_count: number;
  drifted_features_count: number;
  drift_share: number;
  feature_details: FeatureDriftDetail[];
  message?: string;
  last_prediction_at?: string | null;
}

export interface SimulateTrafficResponse {
  status: string;
  simulated_samples: number;
  drift_type: 'normal' | 'drifted';
  message: string;
}

// ─── Treatment Candidates & Validation Types ───────────────────
export interface TreatmentCandidate {
  candidate_id: string;
  name: string;
  strategy_type: string;
  description: string;
  decisions: Array<{
    column?: string | null;
    action: string;
    category?: string;
    parameters?: Record<string, any>;
    status?: string;
  }>;
  summary_tags: string[];
}

export interface TreatmentCandidateEvaluation {
  candidate_id: string;
  name: string;
  strategy_type: string;
  primary_metric_name: string;
  primary_score: number;
  primary_std: number;
  secondary_metric_name: string;
  secondary_score: number;
  rank: number;
  summary_tags: string[];
  training_time_seconds: number;
}

export interface TreatmentValidationReport {
  task_type: string;
  benchmark_model: string;
  primary_metric: string;
  validation_status: 'SUPERIOR' | 'INCONCLUSIVE' | 'SUBOPTIMAL' | 'INVALID_EVALUATION' | string;
  headline: string;
  explanation: string;
  lift: number | null;
  lift_percentage: number | null;
  mean_difference?: number | null;
  relative_difference?: number | null;
  best_candidate_id: string;
  confidence_level?: number;
  ci_lower?: number | null;
  ci_upper?: number | null;
  baseline_fold_scores?: number[];
  recommended_fold_scores?: number[];
  fold_differences?: number[];
  sign_agreement?: string;
  evaluation_metadata?: {
    task_type?: string;
    ci_method?: string;
    degrees_of_freedom?: number | null;
    n_valid_folds?: number;
    estimator?: string;
    estimator_parameters?: Record<string, any>;
    cv_strategy?: string;
    n_splits?: number;
    shuffle?: boolean;
    random_state?: number;
    scoring_metric?: string;
    confidence_level?: number;
  };
  candidates: TreatmentCandidateEvaluation[];
}

export interface ApplyCandidateResponse {
  status: string;
  candidate_id: string;
  applied_count: number;
  decisions: WorkflowDecision[];
}


export interface User {
  id: string;
  email: string;
  username: string;
  role: 'admin' | 'data_scientist' | 'viewer' | string;
  tenant_id?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface Tenant {
  id: string;
  name: string;
  slug: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface RegisterPayload {
  email: string;
  username: string;
  password: string;
  role?: string;
  tenant_id?: string | null;
}

