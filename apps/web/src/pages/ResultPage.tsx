import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Trophy, Download, RefreshCw, AlertCircle, CheckCircle2,
  Clock, BarChart2, ArrowLeft, Cpu, Box, FileText, Code, Activity,
} from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts';
import {
  getProject, getExperiment, getConfusionMatrix, getResiduals, getArtifactDownloadUrl,
  getPythonCodeDownloadUrl, getReportDownloadUrl,
} from '../api/client';
import type { Project, Experiment } from '../types';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getCandidateId(project: Project): string | null {
  if (project.status && project.status.startsWith('candidate:')) {
    return project.status.replace('candidate:', '');
  }
  return null;
}

function getMetric(exp: Experiment, name: string): number | null {
  const m = exp.metrics.find(m => m.name === `${name}_mean` && m.split === 'cv');
  return m ? m.value : null;
}

function fmt(v: number | null, pct = true): string {
  if (v === null) return '—';
  return pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(4);
}

function fmtTime(seconds: number | null): string {
  if (seconds === null) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

// ─── Confusion Matrix Heatmap ──────────────────────────────────────────────────

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  if (!matrix || matrix.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem' }}>
        <AlertCircle size={24} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
        <p>Confusion matrix not available</p>
      </div>
    );
  }

  const maxVal = Math.max(...matrix.flat());
  const labels = matrix.map((_, i) => `Class ${i}`);

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: `80px repeat(${matrix[0].length}, 1fr)`,
        gap: '2px',
        maxWidth: '480px',
      }}>
        <div style={{ gridColumn: 1, gridRow: 1 }} />
        {labels.map((l, j) => (
          <div key={j} style={{
            textAlign: 'center',
            fontSize: '0.72rem',
            color: 'var(--text-muted)',
            padding: '4px 2px',
            fontWeight: 600,
          }}>
            {j === 0 ? <span style={{ fontSize: '0.65rem', opacity: 0.7 }}>Pred→</span> : null}
            {l}
          </div>
        ))}

        {matrix.map((row, i) => (
          <div key={`row-${i}`} style={{ display: 'contents' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              paddingRight: '8px',
              fontWeight: 600,
            }}>
              {i === 0 && <span style={{ fontSize: '0.65rem', opacity: 0.7, marginRight: 4 }}>↓Act.</span>}
              {labels[i]}
            </div>
            {row.map((val, j) => {
              const intensity = maxVal > 0 ? val / maxVal : 0;
              const isDiag = i === j;
              const bg = isDiag
                ? `rgba(99, 102, 241, ${0.2 + intensity * 0.7})`
                : `rgba(239, 68, 68, ${0.05 + intensity * 0.5})`;
              const text = intensity > 0.5 ? '#fff' : 'var(--text-primary)';

              return (
                <div key={`${i}-${j}`} style={{
                  background: bg,
                  color: text,
                  textAlign: 'center',
                  borderRadius: '6px',
                  padding: '12px 4px',
                  fontSize: '1rem',
                  fontWeight: 700,
                  transition: 'all 0.2s',
                  border: isDiag ? '1px solid rgba(99,102,241,0.5)' : 'none',
                }}>
                  {val}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
        <span style={{ marginRight: '1rem' }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'rgba(99,102,241,0.6)', marginRight: 4 }} />
          Correct predictions (diagonal)
        </span>
        <span>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'rgba(239,68,68,0.3)', marginRight: 4 }} />
          Misclassifications
        </span>
      </div>
    </div>
  );
}

// ─── Residual Plot (Regression) ────────────────────────────────────────────────

function ResidualPlot({ data }: { data: { y_true: number; y_pred: number; residual: number }[] }) {
  if (!data || data.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem' }}>
        <AlertCircle size={24} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
        <p>Residual data not available</p>
      </div>
    );
  }

  const minVal = Math.min(...data.map(d => Math.min(d.y_true, d.y_pred)));
  const maxVal = Math.max(...data.map(d => Math.max(d.y_true, d.y_pred)));

  return (
    <div style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="y_true"
            name="Actual"
            stroke="var(--text-muted)"
            domain={[minVal, maxVal]}
          />
          <YAxis
            type="number"
            dataKey="y_pred"
            name="Predicted"
            stroke="var(--text-muted)"
            domain={[minVal, maxVal]}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
            formatter={(value: any) => [Number(value).toFixed(3), 'Value']}
          />
          <ReferenceLine segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]} stroke="#f38ba8" strokeDasharray="5 5" />
          <Scatter name="Actual vs Predicted" data={data} fill="#89b4fa" opacity={0.7} />
        </ScatterChart>
      </ResponsiveContainer>
      <div style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
        Linear Reference Line (Red Dashed) represents Perfect Prediction (Actual = Predicted)
      </div>
    </div>
  );
}

// ─── Metric Card ───────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat-card" style={{ flex: '1 1 140px', minWidth: 130 }}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ fontSize: '1.6rem' }}>{value}</div>
      {sub && <div className="stat-label" style={{ fontSize: '0.7rem', marginTop: '0.2rem' }}>{sub}</div>}
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────────

export function ResultPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const [project, setProject] = useState<Project | null>(null);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);
  const [residualData, setResidualData] = useState<{ y_true: number; y_pred: number; residual: number }[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const proj = await getProject(projectId);
      setProject(proj);

      const candidateId = getCandidateId(proj);
      if (!candidateId) {
        setLoading(false);
        return;
      }

      const [exp, cm, res] = await Promise.all([
        getExperiment(candidateId),
        getConfusionMatrix(candidateId).catch(() => null),
        getResiduals(candidateId).catch(() => null),
      ]);
      setExperiment(exp);
      setConfusionMatrix(cm?.confusion_matrix ?? null);
      setResidualData(res?.residual_data ?? null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load results');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  const handleDownload = (artifactType: string) => {
    if (!experiment) return;
    setDownloading(artifactType);
    const url = getArtifactDownloadUrl(experiment.id, artifactType);
    const a = document.createElement('a');
    a.href = url;
    a.click();
    setTimeout(() => setDownloading(null), 1500);
  };

  // ── Loading ──
  if (loading) {
    return (
      <div style={{ padding: '2rem' }} className="animate-fade-in">
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', color: 'var(--text-muted)' }}>
          <RefreshCw size={16} className="animate-spin" />
          Loading results…
        </div>
      </div>
    );
  }

  // ── No candidate selected ──
  if (!project || !getCandidateId(project)) {
    return (
      <div style={{ padding: '2rem' }} className="animate-fade-in">
        <div className="page-header">
          <Trophy size={22} className="text-gradient" />
          <div>
            <h1 className="page-title">Final Result</h1>
            <p className="page-subtitle">No candidate model selected yet</p>
          </div>
        </div>
        <div className="card" style={{ marginTop: '1.5rem', textAlign: 'center', padding: '3rem' }}>
          <AlertCircle size={40} style={{ marginBottom: '1rem', opacity: 0.4 }} />
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
            Go to the <strong>Experiments</strong> page, run experiments, then click{' '}
            <strong>"Select as Candidate"</strong> on the best model.
          </p>
          <button
            className="btn btn-primary"
            onClick={() => navigate(`/projects/${projectId}/experiments`)}
          >
            <ArrowLeft size={14} /> Go to Experiments
          </button>
        </div>
      </div>
    );
  }

  // ── Error ──
  if (error || !experiment) {
    return (
      <div style={{ padding: '2rem' }} className="animate-fade-in">
        <div className="alert alert-error">
          <AlertCircle size={16} />
          {error ?? 'Experiment data not available'}
        </div>
      </div>
    );
  }

  const isRegression = project.task_type === 'regression';

  const metrics = isRegression ? {
    r2: getMetric(experiment, 'r2'),
    mae: getMetric(experiment, 'mae'),
    mse: getMetric(experiment, 'mse'),
    rmse: getMetric(experiment, 'rmse'),
  } : {
    accuracy: getMetric(experiment, 'accuracy'),
    f1: getMetric(experiment, 'f1'),
    precision: getMetric(experiment, 'precision'),
    recall: getMetric(experiment, 'recall'),
    roc_auc: getMetric(experiment, 'roc_auc'),
    pr_auc: getMetric(experiment, 'pr_auc'),
  };

  const artifactTypes = experiment.artifacts.map(a => a.type);

  return (
    <div style={{ padding: '2rem' }} className="animate-fade-in">
      {/* Header */}
      <div className="page-header">
        <Trophy size={22} style={{ color: '#f59e0b' }} />
        <div>
          <h1 className="page-title">Final Result ({isRegression ? 'Regression' : 'Classification'})</h1>
          <p className="page-subtitle">Selected candidate model &amp; validation summary</p>
        </div>
      </div>

      {/* Selected model banner */}
      <div className="card animate-slide-up" style={{
        marginTop: '1.5rem',
        background: 'linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%)',
        border: '1px solid rgba(99,102,241,0.3)',
        padding: '1.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{
            width: 52, height: 52, borderRadius: 12,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Cpu size={24} color="white" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              Selected Candidate Model
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {experiment.model_name}
            </div>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: 4, alignItems: 'center' }}>
                <Clock size={12} /> {(experiment.config_json?.cv_folds as number) ?? 5}-fold CV · Seed {experiment.random_seed}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: 4, alignItems: 'center' }}>
                <Clock size={12} /> Training time: {fmtTime(experiment.training_time_seconds)}
              </span>
              <span className={`badge badge-${experiment.status === 'completed' ? 'success' : 'error'}`}>
                <CheckCircle2 size={10} /> {experiment.status}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ marginTop: '1.5rem' }}>
        <div className="section-title" style={{ marginBottom: '0.75rem' }}>
          <BarChart2 size={14} /> Validation Metrics (CV Mean)
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {isRegression ? (
            <>
              <MetricCard label="R² Score" value={fmt((metrics as any).r2, false)} />
              <MetricCard label="MAE" value={fmt((metrics as any).mae, false)} />
              <MetricCard label="MSE" value={fmt((metrics as any).mse, false)} />
              <MetricCard label="RMSE" value={fmt((metrics as any).rmse, false)} />
            </>
          ) : (
            <>
              <MetricCard label="Accuracy" value={fmt((metrics as any).accuracy)} />
              <MetricCard label="F1 Score" value={fmt((metrics as any).f1)} />
              <MetricCard label="Precision" value={fmt((metrics as any).precision)} />
              <MetricCard label="Recall" value={fmt((metrics as any).recall)} />
              {(metrics as any).roc_auc !== null && (
                <MetricCard label="ROC-AUC" value={fmt((metrics as any).roc_auc)} />
              )}
              {(metrics as any).pr_auc !== null && (
                <MetricCard label="PR-AUC" value={fmt((metrics as any).pr_auc)} />
              )}
            </>
          )}
        </div>
      </div>

      {/* Pipeline summary */}
      {experiment.config_json && (
        <div className="card animate-slide-up" style={{ marginTop: '1.5rem' }}>
          <div className="section-title" style={{ marginBottom: '1rem' }}>
            <Box size={14} /> Pipeline Configuration
          </div>
          <div style={{
            background: 'var(--surface-3)',
            borderRadius: 8,
            padding: '1rem',
            fontSize: '0.8rem',
            fontFamily: 'monospace',
            color: 'var(--text-secondary)',
            maxHeight: 240,
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {JSON.stringify(experiment.config_json, null, 2)}
          </div>
        </div>
      )}

      {/* Confusion Matrix or Residual Plot */}
      <div className="card animate-slide-up" style={{ marginTop: '1.5rem' }}>
        <div className="section-title" style={{ marginBottom: '1rem' }}>
          {isRegression ? <Activity size={14} /> : <BarChart2 size={14} />} {isRegression ? 'Actual vs Predicted Scatter Plot & Residuals' : 'Confusion Matrix (Aggregate CV Predictions)'}
        </div>
        {isRegression ? (
          <ResidualPlot data={residualData ?? []} />
        ) : (
          <ConfusionMatrix matrix={confusionMatrix ?? []} />
        )}
      </div>

      {/* Warnings */}
      {experiment.status !== 'completed' && (
        <div className="alert alert-warning" style={{ marginTop: '1.5rem' }}>
          <AlertCircle size={14} />
          This experiment did not complete successfully. Export may be unavailable.
        </div>
      )}

      {/* Export */}
      <div className="card animate-slide-up" style={{ marginTop: '1.5rem' }}>
        <div className="section-title" style={{ marginBottom: '1rem' }}>
          <Download size={14} /> Export &amp; Tracking
        </div>

        {/* Generated Code & Report */}
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <a
            className="btn btn-primary"
            href={getPythonCodeDownloadUrl(experiment.id)}
            download
            style={{
              display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
              gap: '0.2rem', padding: '0.75rem 1.25rem', minWidth: 180,
              textDecoration: 'none',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
              <Code size={14} /> Python Script (.py)
            </span>
            <span style={{ fontSize: '0.68rem', opacity: 0.8, fontWeight: 400 }}>
              Reproducible sklearn code
            </span>
          </a>

          {projectId && (
            <a
              className="btn btn-primary"
              href={getReportDownloadUrl(projectId)}
              download
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
                gap: '0.2rem', padding: '0.75rem 1.25rem', minWidth: 180,
                textDecoration: 'none',
                background: 'linear-gradient(135deg, #10b981, #059669)',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                <FileText size={14} /> Project Report (.html)
              </span>
              <span style={{ fontSize: '0.68rem', opacity: 0.8, fontWeight: 400 }}>
                Full summary &amp; CM report
              </span>
            </a>
          )}
        </div>

        {/* Model Artifacts */}
        <div className="section-title" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
          Raw Model Artifacts
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {[
            { type: 'pipeline', label: 'Pipeline (.pkl)', desc: 'Trained sklearn Pipeline' },
            { type: 'config', label: 'Config (.json)', desc: 'Preprocessing configuration' },
            { type: 'metrics', label: 'Metrics (.json)', desc: 'CV metrics per fold' },
          ].map(({ type, label, desc }) => {
            const available = artifactTypes.includes(type);
            return (
              <button
                key={type}
                className={`btn ${available ? 'btn-primary' : 'btn-ghost'}`}
                disabled={!available || downloading === type}
                onClick={() => handleDownload(type)}
                style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
                  gap: '0.2rem', padding: '0.75rem 1.25rem',
                  opacity: available ? 1 : 0.4,
                  minWidth: 160,
                }}
                title={available ? `Download ${label}` : 'Artifact not available'}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                  {downloading === type ? (
                    <RefreshCw size={13} className="animate-spin" />
                  ) : (
                    <Download size={13} />
                  )}
                  {label}
                </span>
                <span style={{ fontSize: '0.68rem', opacity: 0.7, fontWeight: 400 }}>{desc}</span>
              </button>
            );
          })}
        </div>
        {artifactTypes.length === 0 && (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
            No artifacts saved. Artifacts are generated when experiments run successfully.
          </p>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
        <button
          className="btn btn-ghost"
          onClick={() => navigate(`/projects/${projectId}/experiments`)}
        >
          <ArrowLeft size={14} /> Back to Experiments
        </button>
        <button
          className="btn btn-ghost"
          onClick={() => navigate('/')}
        >
          All Projects
        </button>
      </div>
    </div>
  );
}
