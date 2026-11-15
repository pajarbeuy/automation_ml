import { useState, useEffect, Fragment } from 'react';
import { createPortal } from 'react-dom';
import { useParams, useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { FlaskConical, Play, CheckCircle2, XCircle, Clock, Trophy, Sliders, Zap, Sparkles, Search, X } from 'lucide-react';
import { listExperiments, runExperiments, getAvailableModels, selectCandidate, getExperimentSummary, tuneHyperparameters, getProject } from '../api/client';
import type { Experiment, TuningResultData } from '../types';

const MODEL_DISPLAY: Record<string, string> = {
  dummy_classifier: 'Dummy Classifier',
  logistic_regression: 'Logistic Regression',
  linear_svm: 'Linear SVM',
  random_forest: 'Random Forest Classifier',
  gradient_boosting: 'Gradient Boosting Classifier',
  hist_gradient_boosting: 'HistGrad Boost Classifier',
  knn: 'K-Nearest Neighbors Classifier',
  // Regression
  dummy_regressor: 'Dummy Regressor',
  linear_regression: 'Linear Regression',
  ridge: 'Ridge Regression',
  random_forest_regressor: 'Random Forest Regressor',
  gradient_boosting_regressor: 'Gradient Boosting Regressor',
  hist_gradient_boosting_regressor: 'HistGrad Boost Regressor',
  knn_regressor: 'KNN Regressor',
  xgb_regressor: 'XGBoost Regressor',
  lgbm_regressor: 'LightGBM Regressor',
  // Sentiment Analysis & NLP
  multinomial_nb: 'Multinomial Naive Bayes',
  complement_nb: 'Complement Naive Bayes',
  sgd_classifier: 'SGD Classifier',
};

const STATUS_ICON = {
  completed: <CheckCircle2 size={14} color="var(--success)" />,
  failed: <XCircle size={14} color="var(--danger)" />,
  running: <span className="spinner" />,
  pending: <Clock size={14} color="var(--text-muted)" />,
};

function getMetric(exp: Experiment, name: string): number | null {
  const m = exp.metrics.find(m => m.name === name && m.split === 'cv');
  return m ? m.value : null;
}

function renderMarkdownBold(text?: string) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}

export function ExperimentsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [taskType, setTaskType] = useState<string>('classification');
  const [primaryMetric, setPrimaryMetric] = useState('f1_mean');
  const [cvFolds, setCvFolds] = useState(5);
  const [seed, setSeed] = useState(42);
  const [enableFe, setEnableFe] = useState(false);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectingCandidate, setSelectingCandidate] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [aiSummary, setAiSummary] = useState<Record<string, any> | null>(null);
  const [experimentName, setExperimentName] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // State Hyperparameter Tuning
  const [showTuneModal, setShowTuneModal] = useState(false);
  const [tuneModel, setTuneModel] = useState('random_forest');
  const [nTrials, setNTrials] = useState(10);
  const [isTuning, setIsTuning] = useState(false);
  const [tuningResult, setTuningResult] = useState<TuningResultData | null>(null);

  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    getProject(projectId).then(proj => {
      if (proj.task_type) {
        setTaskType(proj.task_type);
        setPrimaryMetric(proj.task_type === 'regression' ? 'r2_mean' : 'f1_mean');
      }
      return Promise.all([
        listExperiments(projectId),
        getAvailableModels(proj.task_type || 'classification'),
        getExperimentSummary(projectId).catch(() => null),
      ]);
    }).then(([exps, models, summary]) => {
      setExperiments(exps);
      setAvailableModels(models);
      setSelectedModels(models.slice(0, 3));
      if (summary) setAiSummary(summary as Record<string, any>);
    }).finally(() => setLoading(false));
  }, [projectId]);

  // Polling otomatis untuk eksperimen yang sedang berjalan & update AI summary saat selesai
  useEffect(() => {
    const hasRunning = experiments.some(e => e.status === 'pending' || e.status === 'running');
    if (!hasRunning || !projectId) return;

    const interval = setInterval(async () => {
      try {
        const updated = await listExperiments(projectId);
        setExperiments(updated);
        const stillRunning = updated.some(e => e.status === 'pending' || e.status === 'running');
        if (!stillRunning) {
          clearInterval(interval);
          getExperimentSummary(projectId).then(s => {
            if (s) setAiSummary(s as Record<string, any>);
          }).catch(() => null);
        }
      } catch {
        // Non-fatal polling error
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [experiments, projectId]);

  const handleRunTuning = async () => {
    if (!projectId) return;
    setIsTuning(true);
    try {
      const res = await tuneHyperparameters(projectId, {
        model_name: tuneModel,
        n_trials: nTrials,
        random_seed: seed,
        cv_folds: cvFolds,
        primary_metric: primaryMetric.replace('_mean', ''),
      });
      setTuningResult(res as TuningResultData);
    } catch {
      alert('Gagal menjalankan Hyperparameter Tuning.');
    } finally {
      setIsTuning(false);
    }
  };

  const handleRun = async () => {
    if (!projectId) return;
    setRunning(true);
    setRunError(null);
    try {
      const newExps = await runExperiments(projectId, {
        model_names: selectedModels,
        experiment_name: experimentName.trim() || undefined,
        random_seed: seed,
        cv_folds: cvFolds,
        primary_metric: primaryMetric.replace('_mean', ''),
        enable_feature_engineering: enableFe,
      });
      setExperiments(prev => [...newExps, ...prev]);
      const allFailed = newExps.every(e => e.status === 'failed');
      if (allFailed && newExps.length > 0) {
        const firstErr = newExps[0]?.error_message ?? 'Unknown error';
        setRunError(`All experiments failed. Reason: ${firstErr}`);
      }
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (err as Error)?.message ?? 'Request failed';
      setRunError(msg);
    } finally {
      setRunning(false);
    }
  };

  const handleSelectCandidate = async (experimentId: string) => {
    if (!projectId) return;
    setSelectingCandidate(experimentId);
    try {
      await selectCandidate(experimentId);
      navigate(`/projects/${projectId}/result`);
    } finally {
      setSelectingCandidate(null);
    }
  };

  const handleOpenTuneModal = (modelName?: string) => {
    if (modelName) {
      setTuneModel(modelName);
    } else if (chartData.length > 0 && chartData[0]?.model_name) {
      setTuneModel(chartData[0].model_name);
    } else if (availableModels.length > 0) {
      setTuneModel(availableModels[0]);
    }
    setTuningResult(null);
    setShowTuneModal(true);
  };

  const handleRunTunedExperiment = async () => {
    if (!projectId || !tuneModel) return;
    setShowTuneModal(false);
    setRunning(true);
    setRunError(null);
    try {
      const newExps = await runExperiments(projectId, {
        model_names: [tuneModel],
        experiment_name: `Tuned: ${MODEL_DISPLAY[tuneModel] ?? tuneModel}`,
        random_seed: seed,
        cv_folds: cvFolds,
        primary_metric: primaryMetric.replace('_mean', ''),
        enable_feature_engineering: enableFe,
        is_tuned: true,
        model_params: tuningResult?.best_params,
      });
      setExperiments(prev => [...newExps, ...prev]);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (err as Error)?.message ?? 'Request failed';
      setRunError(msg);
    } finally {
      setRunning(false);
    }
  };


  // Build comparison chart data with model + experiment name disambiguation
  const metricForChart = primaryMetric;
  const isRegression = taskType === 'regression';

  const chartData = experiments
    .filter(e => e.status === 'completed')
    .map(e => {
      const expName = e.experiment_name || (e.config_json?.experiment_name as string);
      const baseName = MODEL_DISPLAY[e.model_name] ?? e.model_name;
      const label = expName ? `${baseName} (${expName})` : baseName;
      return {
        id: e.id,
        model: label,
        model_name: e.model_name,
        mean: getMetric(e, metricForChart) ?? 0,
        std: getMetric(e, metricForChart.replace('_mean', '_std')) ?? 0,
      };
    })
    .sort((a, b) => isRegression ? b.mean - a.mean : b.mean - a.mean);

  const best = chartData[0];

  // Filtered experiments for search query
  const filteredExperiments = experiments.filter(exp => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    const modelDisplayName = (MODEL_DISPLAY[exp.model_name] ?? exp.model_name).toLowerCase();
    const rawModel = exp.model_name.toLowerCase();
    const expName = ((exp.experiment_name || (exp.config_json?.experiment_name as string)) ?? '').toLowerCase();
    const status = exp.status.toLowerCase();
    return modelDisplayName.includes(q) || rawModel.includes(q) || expName.includes(q) || status.includes(q);
  });

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: '1rem' }}>
      <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      <div style={{ color: 'var(--text-secondary)' }}>Loading experiments…</div>
    </div>
  );

  const metricOptions = isRegression
    ? ['r2_mean', 'mae_mean', 'mse_mean', 'rmse_mean']
    : ['f1_mean', 'accuracy_mean', 'precision_mean', 'recall_mean', 'roc_auc_mean'];

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Experiment Workspace ({isRegression ? 'Regression' : 'Classification'})</h1>
          <p className="page-subtitle">Train, tune hyperparameters, and compare baseline {isRegression ? 'regressors' : 'classifiers'}</p>
        </div>
        <div>
          <button
            className="btn btn-secondary"
            onClick={() => handleOpenTuneModal()}
            style={{ border: '1px solid rgba(137, 180, 250, 0.4)', color: '#89b4fa' }}
          >
            <Sliders size={16} /> Hyperparameter Tuning Engine
          </button>
        </div>
      </div>

      <div className="grid-2" style={{ gap: '1.5rem', alignItems: 'start', marginBottom: '1.5rem' }}>
        {/* Run config */}
        <div className="card">
          <h3 style={{ marginBottom: '1.25rem' }}>Run Configuration</h3>

          <div className="form-group" style={{ marginBottom: '1rem' }}>
            <label className="label">Models to Train</label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem' }}>
              {availableModels.map(m => (
                <button
                  key={m}
                  className={`btn btn-sm ${selectedModels.includes(m) ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => setSelectedModels(prev =>
                    prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
                  )}
                >
                  {MODEL_DISPLAY[m] ?? m}
                </button>
              ))}
            </div>
          </div>

          <div className="grid-2" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
            <div className="form-group">
              <label className="label" htmlFor="primary-metric">Primary Metric</label>
              <select
                id="primary-metric"
                className="input select"
                value={primaryMetric}
                onChange={e => setPrimaryMetric(e.target.value)}
              >
                {metricOptions.map(m => (
                  <option key={m} value={m}>{m.replace('_mean', '').toUpperCase()}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="label" htmlFor="cv-folds">CV Folds</label>
              <select
                id="cv-folds"
                className="input select"
                value={cvFolds}
                onChange={e => setCvFolds(Number(e.target.value))}
              >
                {[3, 5, 10].map(n => <option key={n} value={n}>{n}-fold</option>)}
              </select>
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: '1rem' }}>
            <label className="label">Random Seed</label>
            <input
              className="input"
              type="number"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              style={{ maxWidth: 120 }}
            />
          </div>

          <div className="form-group" style={{ marginBottom: '1rem' }}>
            <label className="label" htmlFor="exp-name-input">
              Experiment / Pipeline Name (Opsional)
            </label>
            <input
              id="exp-name-input"
              className="input"
              type="text"
              placeholder="Misal: Pipeline Treatment Rekomendasi, Outlier Capped, dsb."
              value={experimentName}
              onChange={e => setExperimentName(e.target.value)}
              style={{ width: '100%' }}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem', display: 'block' }}>
              Beri label pipeline ini agar tidak membingungkan saat menguji algoritma model yang sama dengan perlakuan berbeda.
            </span>
          </div>

          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.85rem', color: '#cdd6f4' }}>
              <input
                type="checkbox"
                checked={enableFe}
                onChange={e => setEnableFe(e.target.checked)}
                style={{ accentColor: '#89b4fa', width: 16, height: 16 }}
              />
              <Sparkles size={14} style={{ color: '#f9e2af' }} />
              Auto Feature Engineering (Log Transformation & Interactions)
            </label>
          </div>

          <button
            id="run-experiments-btn"
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={handleRun}
            disabled={running || selectedModels.length === 0}
          >
            {running ? (
              <>
                <span className="spinner" />
                Training models… (this may take a minute)
              </>
            ) : (
              <>
                <Play size={16} />
                Run Experiments ({selectedModels.length} models)
              </>
            )}
          </button>

          {running && (
            <div className="alert alert-info" style={{ marginTop: '0.875rem' }}>
              <Clock size={14} />
              Running {cvFolds}-fold CV for each model. Please wait…
            </div>
          )}

          {runError && (
            <div className="alert alert-error" style={{ marginTop: '0.875rem' }}>
              <XCircle size={14} />
              <div>
                <strong>Experiment Error</strong>
                <div style={{ fontSize: '0.8rem', marginTop: '0.25rem', opacity: 0.9 }}>{runError}</div>
              </div>
            </div>
          )}
        </div>

        {/* Best model card */}
        {best && (
          <div className="card" style={{ borderColor: 'rgba(16,185,129,0.3)', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Trophy size={18} color="var(--warning)" />
                  <h3 style={{ margin: 0 }}>Best Performing Model</h3>
                </div>
                <button
                  className="btn btn-sm btn-secondary"
                  style={{ border: '1px solid rgba(249, 226, 175, 0.5)', color: '#f9e2af', fontSize: '0.78rem', gap: '0.35rem' }}
                  onClick={() => handleOpenTuneModal(best.model_name)}
                  title="Optimize hyperparameters to boost accuracy for the best model"
                >
                  <Sliders size={12} /> Tune Best Model
                </button>
              </div>
              <div className="metric-value" style={{ fontSize: '2rem', marginBottom: '0.25rem' }}>
                {best.model}
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <span className="badge badge-success">
                  {isRegression ? best.mean.toFixed(4) : `${(best.mean * 100).toFixed(2)}%`}
                </span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  ± {isRegression ? best.std.toFixed(4) : `${(best.std * 100).toFixed(2)}%`} (CV std)
                </span>
              </div>
            </div>
          </div>
        )}

        {/* AI Experiment Summary */}
        {aiSummary && (
          <div className="card" style={{
            gridColumn: '1 / -1',
            border: '1px solid var(--border-highlight)',
            padding: '1.5rem',
            boxShadow: 'var(--neu-raised)',
          }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{
                  width: 36,
                  height: 36,
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(67, 56, 202, 0.12)',
                  border: '1px solid rgba(67, 56, 202, 0.25)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent)',
                  flexShrink: 0,
                }}>
                  <Sparkles size={18} />
                </div>
                <div>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent)' }}>
                    AI Benchmark Summary
                  </div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                    {aiSummary.headline}
                  </h3>
                </div>
              </div>

              {aiSummary.best_model_name && (
                <span className="badge badge-primary" style={{ fontSize: '0.75rem', padding: '0.3rem 0.75rem' }}>
                  <Trophy size={13} style={{ marginRight: '0.25rem' }} /> Rekomendasi Utama
                </span>
              )}
            </div>

            {/* Narrative Box */}
            <div style={{
              fontSize: '0.88rem',
              color: 'var(--text-secondary)',
              lineHeight: 1.65,
              padding: '1rem 1.25rem',
              background: 'var(--bg-inset)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-inset)',
              marginBottom: '1rem',
            }}>
              {renderMarkdownBold(aiSummary.narrative)}
            </div>

            {/* Strengths & Takeaways Grid */}
            {((aiSummary.strengths && aiSummary.strengths.length > 0) || (aiSummary.takeaways && aiSummary.takeaways.length > 0)) && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                {aiSummary.strengths && aiSummary.strengths.length > 0 && (
                  <div style={{
                    background: 'rgba(5, 150, 105, 0.06)',
                    border: '1px solid rgba(5, 150, 105, 0.25)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.85rem 1rem',
                  }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--success)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <CheckCircle2 size={13} /> Bukti Performa Model
                    </div>
                    <ul style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                      {aiSummary.strengths.map((str: string, idx: number) => (
                        <li key={idx} style={{ marginBottom: '0.25rem' }}>{str}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {aiSummary.takeaways && aiSummary.takeaways.length > 0 && (
                  <div style={{
                    background: 'rgba(67, 56, 202, 0.06)',
                    border: '1px solid rgba(67, 56, 202, 0.25)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.85rem 1rem',
                  }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Clock size={13} /> Status & Rekomendasi Langkah
                    </div>
                    <ul style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                      {aiSummary.takeaways.map((tak: string, idx: number) => (
                        <li key={idx} style={{ marginBottom: '0.25rem' }}>{tak}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* MODAL HYPERPARAMETER TUNING */}
      {showTuneModal && createPortal(
        <div
          onClick={() => setShowTuneModal(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 10000,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1.5rem',
            backdropFilter: 'blur(4px)',
          }}
        >
          <div
            className="animate-slide-up card"
            onClick={e => e.stopPropagation()}
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: '16px',
              maxWidth: '750px',
              width: '100%',
              padding: '1.75rem',
              color: 'var(--text-primary)',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: 'var(--neu-raised-hover)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-inset)', paddingBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Zap size={22} style={{ color: 'var(--accent)' }} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                  Hyperparameter Optimization Engine (Randomized Search)
                </h3>
              </div>
              <button onClick={() => setShowTuneModal(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.25rem' }}>
                &times;
              </button>
            </div>

            <div className="grid-2" style={{ gap: '1rem', marginBottom: '1.25rem', background: 'var(--bg-inset)', padding: '1rem', borderRadius: '10px', boxShadow: 'var(--neu-inset-sm)' }}>
              <div className="form-group">
                <label className="label">Pilih Algoritma Model:</label>
                <select
                  value={tuneModel}
                  onChange={e => setTuneModel(e.target.value)}
                  className="select input"
                  style={{ width: '100%' }}
                >
                  {availableModels.map(m => (
                    <option key={m} value={m}>{MODEL_DISPLAY[m] ?? m}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="label">Jumlah Trials (Pencarian Kombinasi):</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <input
                    type="range"
                    min={5}
                    max={30}
                    step={5}
                    value={nTrials}
                    onChange={e => setNTrials(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <strong style={{ fontSize: '0.9rem', color: 'var(--accent)' }}>{nTrials} Trials</strong>
                </div>
              </div>
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginBottom: '1.25rem' }}
              onClick={handleRunTuning}
              disabled={isTuning}
            >
              {isTuning ? (
                <>
                  <span className="spinner" />
                  Running Search Trials (CV)...
                </>
              ) : (
                <>
                  <Zap size={16} />
                  Start Hyperparameter Optimization
                </>
              )}
            </button>

            {/* TUNING RESULTS */}
            {tuningResult && (
              <div className="animate-fade-in" style={{ background: 'var(--bg-inset)', borderRadius: '12px', padding: '1.25rem', border: '1px solid var(--border-inset)', boxShadow: 'var(--neu-inset-sm)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div>
                    <h4 style={{ margin: 0, color: 'var(--success)', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Trophy size={18} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                      Skor Terbaik: {tuningResult.best_score.toFixed(4)} ({tuningResult.primary_metric})
                    </h4>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Waktu Optimasi: {tuningResult.total_time_seconds}s | Trials: {tuningResult.trials.length}</span>
                  </div>
                  <button
                    className="btn btn-primary"
                    style={{ fontWeight: 600, fontSize: '0.82rem', gap: '0.35rem' }}
                    onClick={handleRunTunedExperiment}
                  >
                    <Play size={14} /> Run & Register Tuned Model
                  </button>
                </div>

                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent)', marginBottom: '0.5rem' }}>
                  Tuned Hyperparameters Terpilih:
                </div>

                <div style={{ background: 'var(--bg-card)', borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem', border: '1px solid var(--border)', boxShadow: 'var(--neu-raised-sm)' }}>
                  <table style={{ width: '100%', fontSize: '0.82rem', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-inset)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                        <th style={{ padding: '0.3rem' }}>Parameter</th>
                        <th style={{ padding: '0.3rem' }}>Nilai Hasil Tuning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(tuningResult.best_params).map(([param, val]) => (
                        <tr key={param} style={{ borderBottom: '1px solid var(--border-inset)' }}>
                          <td style={{ padding: '0.3rem', color: 'var(--accent)', fontFamily: 'monospace', fontWeight: 600 }}>{param}</td>
                          <td style={{ padding: '0.3rem', fontWeight: 600, color: 'var(--success)' }}>{String(val)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                  Riwayat Progres Trials:
                </div>
                <div style={{ maxHeight: '150px', overflowY: 'auto', background: 'var(--bg-inset)', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border-inset)', boxShadow: 'var(--neu-inset-sm)' }}>
                  {tuningResult.trials.map(t => (
                    <div key={t.trial_index} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', padding: '0.3rem 0.5rem', borderBottom: '1px solid var(--border-inset)' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Trial #{t.trial_index}</span>
                      <span>Score: <strong style={{ color: 'var(--success)' }}>{t.score.toFixed(4)}</strong></span>
                      <span style={{ color: 'var(--text-muted)' }}>{t.duration_seconds}s</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ textAlign: 'right', marginTop: '1.5rem' }}>
              <button className="btn btn-primary" onClick={() => setShowTuneModal(false)}>
                Selesai
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Comparison chart */}
      {chartData.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ marginBottom: '1.25rem' }}>Model Comparison — {metricForChart}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ bottom: 30 }}>
              <XAxis
                dataKey="model"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                angle={-25}
                textAnchor="end"
              />
              <YAxis
                stroke="var(--text-muted)"
                tickFormatter={v => isRegression && !primaryMetric.includes('r2') ? Number(v).toFixed(2) : `${(v * 100).toFixed(0)}%`}
                domain={isRegression && !primaryMetric.includes('r2') ? ['auto', 'auto'] : [0, 1]}
              />
              <Tooltip
                formatter={(v: any) => isRegression && !primaryMetric.includes('r2') ? Number(v).toFixed(4) : `${(Number(v) * 100).toFixed(2)}%`}
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
              />
              <Bar dataKey="mean" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Experiments table */}
      {experiments.length > 0 && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <h3 style={{ margin: 0 }}>All Experiments</h3>
              <span className="badge badge-secondary" style={{ fontSize: '0.75rem' }}>
                {filteredExperiments.length} {filteredExperiments.length === experiments.length ? 'runs' : `dari ${experiments.length} runs`}
              </span>
            </div>

            {/* Fitur Search Experiment Name / Algorithm */}
            <div style={{ position: 'relative', minWidth: 280 }}>
              <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                id="search-experiments-input"
                type="text"
                className="input"
                placeholder="Cari algoritma, nama eksperimen, status..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                style={{ paddingLeft: '2rem', paddingRight: searchQuery ? '2rem' : '0.75rem', width: '100%', fontSize: '0.82rem', height: 34 }}
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}
                  title="Bersihkan pencarian"
                >
                  <X size={14} />
                </button>
              )}
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Model &amp; Pipeline / Experiment</th>
                  <th>{isRegression ? 'R² CV (mean±std)' : 'F1 CV (mean±std)'}</th>
                  <th>{isRegression ? 'MAE' : 'Accuracy'}</th>
                  <th>{isRegression ? 'RMSE' : 'ROC-AUC'}</th>
                  <th>Train Time</th>
                  <th>Seed</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredExperiments.length === 0 && (
                  <tr>
                    <td colSpan={8} style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
                      Tidak ada eksperimen yang cocok dengan pencarian "{searchQuery}".
                    </td>
                  </tr>
                )}
                {filteredExperiments.map((exp) => {
                  const m1 = getMetric(exp, isRegression ? 'r2_mean' : 'f1_mean');
                  const m1std = getMetric(exp, isRegression ? 'r2_std' : 'f1_std');
                  const m2 = getMetric(exp, isRegression ? 'mae_mean' : 'accuracy_mean');
                  const m3 = getMetric(exp, isRegression ? 'rmse_mean' : 'roc_auc_mean');
                  const isBest = chartData[0]?.id === exp.id;
                  const expTag = exp.experiment_name || (exp.config_json?.experiment_name as string) || '';
                  const errorMsg = exp.error_message ?? '';
                  const errorShort = errorMsg.length > 80 ? errorMsg.slice(0, 80) + '\u2026' : errorMsg;
                  return (
                    <Fragment key={exp.id}>
                      <tr style={{
                        ...(isBest && exp.status === 'completed' ? { background: 'rgba(16,185,129,0.05)' } : {}),
                        ...(exp.status === 'failed' ? { opacity: 0.8 } : {}),
                      }}>
                        <td>{STATUS_ICON[exp.status as keyof typeof STATUS_ICON] ?? exp.status}</td>
                        <td style={{ fontWeight: 500 }}>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                              <span style={{ color: 'var(--text-primary)' }}>{MODEL_DISPLAY[exp.model_name] ?? exp.model_name}</span>
                              {(exp.config_json?.is_tuned === true || exp.model_name.endsWith('_tune')) && (
                                <span className="badge badge-info" style={{ fontSize: '0.68rem', padding: '0.08rem 0.35rem' }}>Tuned</span>
                              )}
                              {isBest && exp.status === 'completed' && (
                                <Trophy size={12} color="var(--warning)" />
                              )}
                            </div>
                            {/* Output Nama Algoritma + Experiment Name */}
                            {expTag && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                <span
                                  className="badge badge-primary"
                                  style={{
                                    fontSize: '0.72rem',
                                    padding: '0.1rem 0.45rem',
                                    fontWeight: 600,
                                    letterSpacing: '0.01em',
                                    background: 'rgba(67, 56, 202, 0.18)',
                                    color: '#818cf8',
                                    border: '1px solid rgba(129, 140, 248, 0.3)',
                                  }}
                                >
                                  🏷️ {expTag}
                                </span>
                              </div>
                            )}
                          </div>
                        </td>
                        <td style={{ fontFamily: 'monospace' }}>
                          {m1 !== null ? (
                            <>
                              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{isRegression ? m1.toFixed(4) : `${(m1 * 100).toFixed(2)}%`}</span>
                              {m1std !== null && <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}> ±{isRegression ? m1std.toFixed(4) : `${(m1std * 100).toFixed(2)}%`}</span>}
                            </>
                          ) : exp.status === 'failed' ? (
                            <span
                              className="badge badge-danger"
                              title={errorMsg}
                              style={{ cursor: 'help', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}
                            >
                              Error
                            </span>
                          ) : '\u2014'}
                        </td>
                        <td style={{ color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                          {m2 !== null ? (isRegression ? m2.toFixed(4) : `${(m2 * 100).toFixed(2)}%`) : '—'}
                        </td>
                        <td style={{ color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                          {m3 !== null ? (isRegression ? m3.toFixed(4) : `${(m3 * 100).toFixed(2)}%`) : '—'}
                        </td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                          {exp.training_time_seconds ? `${exp.training_time_seconds.toFixed(1)}s` : '—'}
                        </td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                          {exp.random_seed}
                        </td>
                        <td>
                          {exp.status === 'completed' && (
                            <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                              <button
                                className="btn btn-sm btn-secondary"
                                style={{ border: '1px solid rgba(137, 180, 250, 0.4)', color: '#89b4fa', fontSize: '0.78rem', padding: '0.25rem 0.6rem', gap: '0.25rem' }}
                                onClick={() => handleOpenTuneModal(exp.model_name)}
                                title="Tune hyperparameters to boost accuracy for this model"
                              >
                                <Sliders size={12} /> Tune
                              </button>
                              <button
                                className="btn btn-sm btn-primary"
                                style={{ gap: '0.35rem', whiteSpace: 'nowrap', padding: '0.25rem 0.6rem' }}
                                disabled={selectingCandidate === exp.id}
                                onClick={() => handleSelectCandidate(exp.id)}
                              >
                                {selectingCandidate === exp.id ? (
                                  <span className="spinner" />
                                ) : (
                                  <>
                                    <Trophy size={11} /> Candidate
                                  </>
                                )}
                              </button>
                            </div>
                          )}
                        </td>
                      </tr>
                      {exp.status === 'failed' && errorMsg && (
                        <tr key={`${exp.id}-err`} style={{ background: 'rgba(239,68,68,0.04)' }}>
                          <td />
                          <td colSpan={7} style={{
                            fontSize: '0.75rem',
                            color: 'var(--danger)',
                            padding: '0.35rem 0.75rem 0.6rem',
                            fontFamily: 'monospace',
                            lineHeight: 1.4,
                          }}>
                            {errorShort}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

        </div>
      )}

      {experiments.length === 0 && !running && (
        <div className="empty-state">
          <div className="empty-state-icon"><FlaskConical size={48} strokeWidth={1} /></div>
          <h3>No experiments yet</h3>
          <p>Configure and run experiments above to compare models.</p>
        </div>
      )}
    </div>
  );
}
