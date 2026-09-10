import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  GitBranch, Play, CheckCircle2, AlertTriangle, ChevronRight, RefreshCw,
  FileSpreadsheet, Wrench, Binary, Sliders, Target, Cpu, Cog, XCircle
} from 'lucide-react';
import { generatePipeline, getPipeline, validatePipeline } from '../api/client';
import type { PipelineConfig, PipelineValidation } from '../types';

function getStepIcon(label: string) {
  switch (label) {
    case 'Raw': return <FileSpreadsheet size={18} style={{ color: 'var(--text-secondary)' }} />;
    case 'Imputation': return <Wrench size={18} style={{ color: 'var(--warning)' }} />;
    case 'Encoding': return <Binary size={18} style={{ color: 'var(--info)' }} />;
    case 'Scaling': return <Sliders size={18} style={{ color: 'var(--accent-hover)' }} />;
    case 'Feature selection': return <Target size={18} style={{ color: 'var(--accent)' }} />;
    case 'Model': return <Cpu size={18} style={{ color: 'var(--success)' }} />;
    default: return <Cog size={18} style={{ color: 'var(--text-muted)' }} />;
  }
}

function PipelineNode({ label, sublabel, color }: { label: string; sublabel?: string; color?: string }) {
  return (
    <div className="pipeline-node" style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 34, height: 34, borderRadius: 8, background: 'var(--bg-inset)', boxShadow: 'var(--neu-inset-sm)', flexShrink: 0 }}>
        {getStepIcon(label)}
      </div>
      <div>
        <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{label}</div>
        {sublabel && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{sublabel}</div>}
      </div>
    </div>
  );
}

export function PipelinePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [pipeline, setPipeline] = useState<PipelineConfig | null>(null);
  const [validation, setValidation] = useState<PipelineValidation | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    if (!projectId) return;
    getPipeline(projectId)
      .then(setPipeline)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [projectId]);

  const handleGenerate = async () => {
    if (!projectId) return;
    setGenerating(true);
    try {
      const pc = await generatePipeline(projectId);
      setPipeline(pc);
      setValidation(null);
    } finally {
      setGenerating(false);
    }
  };

  const handleValidate = async () => {
    if (!projectId) return;
    setValidating(true);
    try {
      const v = await validatePipeline(projectId);
      setValidation(v);
    } finally {
      setValidating(false);
    }
  };

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: '1rem' }}>
      <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      <div style={{ color: 'var(--text-secondary)' }}>Loading pipeline…</div>
    </div>
  );

  const cfg = pipeline?.config_json;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Pipeline Workspace</h1>
          <p className="page-subtitle">Preview and validate your leakage-safe sklearn pipeline</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-secondary"
            onClick={handleGenerate}
            disabled={generating}
          >
            {generating ? <span className="spinner" /> : <RefreshCw size={16} />}
            {pipeline ? 'Regenerate' : 'Generate Pipeline'}
          </button>
          {pipeline && (
            <>
              <button
                className="btn btn-secondary"
                onClick={handleValidate}
                disabled={validating}
              >
                {validating ? <span className="spinner" /> : <CheckCircle2 size={16} />}
                Validate
              </button>
              <button
                className="btn btn-primary"
                onClick={() => navigate(`/projects/${projectId}/experiments`)}
              >
                Run Experiments <ChevronRight size={16} />
              </button>
            </>
          )}
        </div>
      </div>

      {!pipeline ? (
        <div className="empty-state">
          <div className="empty-state-icon"><GitBranch size={48} strokeWidth={1} /></div>
          <h3>No pipeline yet</h3>
          <p style={{ marginBottom: '1.5rem' }}>
            Generate a pipeline from your approved preprocessing decisions.
          </p>
          <button className="btn btn-primary" onClick={handleGenerate} disabled={generating}>
            {generating ? <span className="spinner" /> : <Play size={16} />}
            Generate Pipeline
          </button>
        </div>
      ) : (
        <div className="grid-2" style={{ gap: '1.5rem', alignItems: 'start' }}>
          {/* Visual pipeline */}
          <div className="card">
            <h3 style={{ marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <GitBranch size={18} color="var(--accent)" />
              Pipeline Graph
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <PipelineNode label="Raw" sublabel="Input features" />
              <div className="pipeline-arrow">↓</div>
              {(cfg?.numeric_columns?.length ?? 0) > 0 && (
                <>
                  <PipelineNode
                    label="Imputation"
                    sublabel={`${cfg?.numeric_columns?.length} numeric · ${cfg?.categorical_columns?.length} categorical`}
                    color="var(--accent)"
                  />
                  <div className="pipeline-arrow">↓</div>
                </>
              )}
              {(cfg?.categorical_columns?.length ?? 0) > 0 && (
                <>
                  <PipelineNode
                    label="Encoding"
                    sublabel={`${cfg?.categorical_columns?.join(', ')}`}
                    color="var(--info)"
                  />
                  <div className="pipeline-arrow">↓</div>
                </>
              )}
              <PipelineNode
                label="Scaling"
                sublabel="StandardScaler / RobustScaler / MinMaxScaler"
                color="var(--warning)"
              />
              <div className="pipeline-arrow">↓</div>
              <PipelineNode
                label="Model"
                sublabel={cfg?.model?.name ?? 'sklearn estimator'}
                color="var(--success)"
              />
            </div>

            {pipeline.pipeline_hash && (
              <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-elevated)', borderRadius: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Pipeline hash: <code style={{ color: 'var(--accent-hover)' }}>{pipeline.pipeline_hash}</code>
                <br />
                Updated: {new Date(pipeline.updated_at).toLocaleString()}
              </div>
            )}
          </div>

          {/* Config details */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Validation result */}
            {validation && (
              <div className={`alert ${validation.valid ? 'alert-success' : 'alert-danger'}`}>
                {validation.valid ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
                <div>
                  <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                    {validation.valid ? 'Pipeline is valid' : 'Validation failed'}
                  </div>
                  {validation.warnings.map(w => (
                    <div key={w} style={{ fontSize: '0.8rem', opacity: 0.9, display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.2rem' }}>
                      <AlertTriangle size={13} style={{ flexShrink: 0 }} /> {w}
                    </div>
                  ))}
                  {validation.errors.map(e => (
                    <div key={e} style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.2rem' }}>
                      <XCircle size={13} style={{ flexShrink: 0 }} /> {e}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Numeric features */}
            {cfg && Object.keys(cfg.numeric_config ?? {}).length > 0 && (
              <div className="card">
                <h4 style={{ marginBottom: '0.875rem' }}>Numeric Features</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {Object.entries(cfg.numeric_config).map(([col, c]) => (
                    <div key={col} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', padding: '0.5rem', background: 'var(--bg-elevated)', borderRadius: 6 }}>
                      <code style={{ color: 'var(--accent-hover)' }}>{col}</code>
                      <div style={{ display: 'flex', gap: '0.375rem' }}>
                        <span className="badge badge-muted">{c.missing}</span>
                        {c.outlier !== 'leave_unchanged' && <span className="badge badge-warning">{c.outlier}</span>}
                        <span className="badge badge-info">{c.scaling}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Categorical features */}
            {cfg && Object.keys(cfg.categorical_config ?? {}).length > 0 && (
              <div className="card">
                <h4 style={{ marginBottom: '0.875rem' }}>Categorical Features</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {Object.entries(cfg.categorical_config).map(([col, c]) => (
                    <div key={col} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', padding: '0.5rem', background: 'var(--bg-elevated)', borderRadius: 6 }}>
                      <code style={{ color: 'var(--accent-hover)' }}>{col}</code>
                      <div style={{ display: 'flex', gap: '0.375rem' }}>
                        <span className="badge badge-muted">{c.missing}</span>
                        <span className="badge badge-primary">{c.encoding}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Dropped columns */}
            {cfg?.dropped_columns && cfg.dropped_columns.length > 0 && (
              <div className="alert alert-warning">
                <AlertTriangle size={16} />
                <div>
                  <strong>Dropped columns:</strong> {cfg.dropped_columns.join(', ')}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
