import axios from 'axios';
import type {
  Project,
  Dataset,
  DatasetProfile,
  DatasetEDA,
  Recommendation,
  WorkflowDecision,
  PipelineConfig,
  PipelineValidation,
  Experiment,
  ExperimentCreate,
  CandidateSchemaResponse,
  PredictResponse,
  DriftReportResponse,
  SimulateTrafficResponse,
  TreatmentCandidate,
  TreatmentValidationReport,
  ApplyCandidateResponse,
  User,
  AuthResponse,
  LoginPayload,
  RegisterPayload,
} from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT Bearer Token if present in localStorage
api.interceptors.request.use(config => {
  const token = localStorage.getItem('automl_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Helper function to extract human-readable error messages from Axios / FastAPI
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;
    if (data?.detail) {
      if (typeof data.detail === 'string') return data.detail;
      if (Array.isArray(data.detail)) {
        return data.detail
          .map((item: any) => item.msg || item.message || JSON.stringify(item))
          .join(', ');
      }
    }
    if (data?.message) return String(data.message);
    if (error.message) return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'Terjadi kesalahan pada sistem.';
}

api.interceptors.response.use(
  response => response,
  error => {
    const msg = getErrorMessage(error);
    if (error && typeof error === 'object') {
      (error as any).userMessage = msg;
    }
    return Promise.reject(error);
  }
);

// ─── Authentication & Multi-Tenancy ──────────────────────────
export const authApi = {
  login: (payload: LoginPayload) =>
    api.post<AuthResponse>('/api/auth/login', payload).then(r => r.data),
  register: (payload: RegisterPayload) =>
    api.post<User>('/api/auth/register', payload).then(r => r.data),
  getMe: () =>
    api.get<User>('/api/auth/me').then(r => r.data),
  createApiKey: (name: string, expiresInDays?: number) =>
    api.post('/api/auth/api-keys', { name, expires_in_days: expiresInDays }).then(r => r.data),
};

// ─── Projects ────────────────────────────────────────────────
export const createProject = (data: { name: string; description?: string; task_type?: string }) =>
  api.post<Project>('/api/projects', data).then(r => r.data);

export const listProjects = () =>
  api.get<Project[]>('/api/projects').then(r => r.data);

export const getProject = (id: string) =>
  api.get<Project>(`/api/projects/${id}`).then(r => r.data);

export const updateProject = (id: string, data: Partial<Project>) =>
  api.patch<Project>(`/api/projects/${id}`, data).then(r => r.data);

export const deleteProject = (id: string) =>
  api.delete(`/api/projects/${id}`);

// ─── Datasets ────────────────────────────────────────────────
export interface UploadProgressEvent {
  loaded: number;
  total: number;
  percent: number;
  speedBytesPerSec: number;
  etaSeconds: number;
}

export const uploadDataset = (
  projectId: string,
  file: File,
  targetColumn: string,
  onProgress?: (progress: UploadProgressEvent) => void,
  cancelSignal?: AbortSignal,
) => {
  const form = new FormData();
  form.append('file', file);
  form.append('target_column', targetColumn);

  const startTime = Date.now();
  let lastLoaded = 0;
  let lastTime = startTime;

  return api.post<Dataset>(`/api/projects/${projectId}/datasets`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    signal: cancelSignal,
    onUploadProgress: (progressEvent) => {
      const now = Date.now();
      const total = progressEvent.total || file.size || 1;
      const loaded = progressEvent.loaded || 0;
      const percent = Math.min(100, Math.round((loaded * 100) / total));

      const timeDelta = (now - lastTime) / 1000;
      const bytesDelta = loaded - lastLoaded;
      const speed = timeDelta > 0.1 && bytesDelta > 0
        ? bytesDelta / timeDelta
        : (loaded / Math.max((now - startTime) / 1000, 0.05));

      const remainingBytes = Math.max(0, total - loaded);
      const eta = speed > 0 ? remainingBytes / speed : 0;

      if (onProgress) {
        onProgress({
          loaded,
          total,
          percent,
          speedBytesPerSec: speed,
          etaSeconds: Math.max(0, eta),
        });
      }

      if (timeDelta > 0.1) {
        lastLoaded = loaded;
        lastTime = now;
      }
    },
  }).then(r => r.data);
};

export const listDatasets = (projectId: string) =>
  api.get<Dataset[]>(`/api/projects/${projectId}/datasets`).then(r => r.data);

export const getProfile = (datasetId: string) =>
  api.get<DatasetProfile>(`/api/datasets/${datasetId}/profile`).then(r => r.data);

export const getEDA = (datasetId: string) =>
  api.get<DatasetEDA>(`/api/datasets/${datasetId}/eda`).then(r => r.data);

// ─── Recommendations ─────────────────────────────────────────
export const getRecommendations = (datasetId: string) =>
  api.get<Recommendation[]>(`/api/datasets/${datasetId}/recommendations`).then(r => r.data);

export const submitDecision = (
  projectId: string,
  data: {
    recommendation_id: string;
    category: string;
    column_name?: string;
    selected_action: string;
    status: string;
    rationale?: string;
  }
) => api.post<WorkflowDecision>(`/api/projects/${projectId}/decisions`, data).then(r => r.data);

export const listDecisions = (projectId: string) =>
  api.get<WorkflowDecision[]>(`/api/projects/${projectId}/decisions`).then(r => r.data);

// ─── Treatment Candidates & Empirical Validation ──────────────
export const getTreatmentCandidates = (projectId: string) =>
  api.get<TreatmentCandidate[]>(`/api/projects/${projectId}/treatment-candidates`).then(r => r.data);

export const evaluateTreatmentCandidates = (projectId: string) =>
  api.post<TreatmentValidationReport>(`/api/projects/${projectId}/treatment-candidates/evaluate`).then(r => r.data);

export const applyTreatmentCandidate = (projectId: string, candidateId: string) =>
  api.post<ApplyCandidateResponse>(`/api/projects/${projectId}/treatment-candidates/apply`, { candidate_id: candidateId }).then(r => r.data);

// ─── Pipelines ───────────────────────────────────────────────
export const generatePipeline = (projectId: string) =>
  api.post<PipelineConfig>(`/api/projects/${projectId}/pipeline`).then(r => r.data);

export const getPipeline = (projectId: string) =>
  api.get<PipelineConfig>(`/api/projects/${projectId}/pipeline`).then(r => r.data);

export const validatePipeline = (projectId: string) =>
  api.post<PipelineValidation>(`/api/projects/${projectId}/pipeline/validate`).then(r => r.data);

export const getScalingPreview = (projectId: string) =>
  api.get<Record<string, any>>(`/api/projects/${projectId}/scaling-preview`).then(r => r.data);

// ─── Experiments ─────────────────────────────────────────────
export const getAvailableModels = (taskType = 'classification') =>
  api.get<string[]>('/api/models', { params: { task_type: taskType } }).then(r => r.data);

export const runExperiments = (projectId: string, data: ExperimentCreate) =>
  api.post<Experiment[]>(`/api/projects/${projectId}/experiments`, data).then(r => r.data);

export const tuneHyperparameters = (
  projectId: string,
  data: { model_name: string; n_trials?: number; random_seed?: number; cv_folds?: number; primary_metric?: string }
) => api.post(`/api/projects/${projectId}/experiments/tune`, data).then(r => r.data);

export const listExperiments = (projectId: string) =>
  api.get<Experiment[]>(`/api/projects/${projectId}/experiments`).then(r => r.data);


export const getExperiment = (experimentId: string) =>
  api.get<Experiment>(`/api/experiments/${experimentId}`).then(r => r.data);

export const getConfusionMatrix = (experimentId: string) =>
  api.get<{ confusion_matrix: number[][] }>(`/api/experiments/${experimentId}/confusion-matrix`).then(r => r.data);

export const getResiduals = (experimentId: string) =>
  api.get<{ residual_data: { y_true: number; y_pred: number; residual: number }[] }>(
    `/api/experiments/${experimentId}/residuals`
  ).then(r => r.data);

export const selectCandidate = (experimentId: string) =>
  api.post<{ experiment_id: string; project_id: string; message: string }>(
    `/api/experiments/${experimentId}/select-candidate`
  ).then(r => r.data);

export const getArtifactDownloadUrl = (experimentId: string, artifactType: string) =>
  `${import.meta.env.VITE_API_BASE_URL ?? ''}/api/experiments/${experimentId}/download/${artifactType}`;

export const getPythonCodeDownloadUrl = (experimentId: string) =>
  `${import.meta.env.VITE_API_BASE_URL ?? ''}/api/experiments/${experimentId}/code`;

export const getReportDownloadUrl = (projectId: string) =>
  `${import.meta.env.VITE_API_BASE_URL ?? ''}/api/projects/${projectId}/report`;

// ─── AI Assistant ─────────────────────────────────────────────
export const explainRecommendation = (recData: Record<string, unknown>) =>
  api.post('/api/assistant/explain-recommendation', recData).then(r => r.data);

export const getExperimentSummary = (projectId: string) =>
  api.get(`/api/projects/${projectId}/assistant/summary`).then(r => r.data);

export const getNextStepGuidance = (projectId: string, currentPage = 'upload') =>
  api.get(`/api/projects/${projectId}/assistant/next-step`, { params: { current_page: currentPage } }).then(r => r.data);

// ─── Model Serving & Live Prediction ───────────────────────────
export const getCandidateSchema = (projectId: string) =>
  api.get<CandidateSchemaResponse>(`/api/projects/${projectId}/models/candidate/schema`).then(r => r.data);

export const predictSingle = (projectId: string, features: Record<string, any>) =>
  api.post<PredictResponse>(`/api/projects/${projectId}/predict`, { features }).then(r => r.data);

export const predictBatchCsv = (projectId: string, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post(`/api/projects/${projectId}/predict/batch-csv`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  }).then(r => r.data);
};

// ─── Model Drift Monitoring ──────────────────────────────────
export const getProjectDrift = (projectId: string) =>
  api.get<DriftReportResponse>(`/api/projects/${projectId}/drift`).then(r => r.data);

export const simulateDriftTraffic = (projectId: string, driftType: 'normal' | 'drifted' = 'normal', samples = 30) =>
  api.post<SimulateTrafficResponse>(`/api/projects/${projectId}/drift/simulate`, { drift_type: driftType, samples }).then(r => r.data);

export const clearDriftLogs = (projectId: string) =>
  api.delete<{ status: string; deleted_count: number; message: string }>(`/api/projects/${projectId}/drift/logs`).then(r => r.data);

export default api;
