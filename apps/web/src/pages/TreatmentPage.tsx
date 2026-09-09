import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useParams, useNavigate } from 'react-router-dom';
import {
  CheckCircle2, X, RotateCcw, ChevronRight, Info, Sparkles, Bot, BarChart2
} from 'lucide-react';
import { listDatasets, getRecommendations, submitDecision, listDecisions, explainRecommendation, getScalingPreview } from '../api/client';
import type { Recommendation, WorkflowDecision, ScalingPreviewData } from '../types';

const CATEGORY_LABELS: Record<string, string> = {
  missing_value: 'Missing Values',
  outlier: 'Outliers',
  encoding: 'Encoding',
  scaling: 'Scaling',
  class_imbalance: 'Class Imbalance',
};

const CATEGORY_COLORS: Record<string, string> = {
  missing_value: 'badge-warning',
  outlier: 'badge-danger',
  encoding: 'badge-primary',
  scaling: 'badge-info',
  class_imbalance: 'badge-warning',
};

function ActionButton({
  action, selected, onClick,
}: { action: string; selected: boolean; onClick: () => void }) {
  return (
    <button
      className={`btn btn-sm ${selected ? 'btn-primary' : 'btn-secondary'}`}
      onClick={onClick}
      style={{ fontSize: '0.78rem' }}
    >
      {action.replace(/_/g, ' ')}
    </button>
  );
}

function ScoreBar({ score }: { score: number }) {
  const cls = score >= 70 ? 'high' : score >= 50 ? 'medium' : 'low';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <div className="score-bar" style={{ flex: 1 }}>
        <div className={`score-fill ${cls}`} style={{ width: `${score}%` }} />
      </div>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', minWidth: 30 }}>
        {score.toFixed(0)}
      </span>
    </div>
  );
}

function HistogramSvg({ counts, color, minVal, maxVal }: { counts: number[]; color: string; minVal?: number; maxVal?: number }) {
  const max = Math.max(...counts, 1);
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '90px', background: 'rgba(0,0,0,0.2)', padding: '6px', borderRadius: '6px' }}>
        {counts.map((c, idx) => {
          const heightPct = (c / max) * 100;
          return (
            <div
              key={idx}
              title={`Count: ${c}`}
              style={{
                flex: 1,
                height: `${Math.max(4, heightPct)}%`,
                background: color,
                borderRadius: '2px 2px 0 0',
                transition: 'height 0.3s ease',
              }}
            />
          );
        })}
      </div>
      {minVal !== undefined && maxVal !== undefined && (
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#a6adc8', marginTop: '6px', padding: '0 2px', fontFamily: 'monospace' }}>
          <span style={{ color: color, fontWeight: 600 }}>Min: {minVal.toFixed(2)}</span>
          <span style={{ fontSize: '0.65rem', opacity: 0.6 }}>Rentang Nilai Sumbu X</span>
          <span style={{ color: color, fontWeight: 600 }}>Max: {maxVal.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

export function TreatmentPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [decisions, setDecisions] = useState<Record<string, WorkflowDecision>>({});
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [filterCat, setFilterCat] = useState<string>('all');
  const [aiExplanation, setAiExplanation] = useState<Record<string, any> | null>(null);
  const [loadingAi, setLoadingAi] = useState<string | null>(null);

  // State Visualisasi Scaling Before vs After
  const [scalingData, setScalingData] = useState<ScalingPreviewData | null>(null);
  const [showScalingModal, setShowScalingModal] = useState(false);
  const [selectedCol, setSelectedCol] = useState<string | null>(null);
  const [loadingScaling, setLoadingScaling] = useState(false);

  const handleFetchScalingPreview = async () => {
    if (!projectId) return;
    setLoadingScaling(true);
    try {
      const data = await getScalingPreview(projectId);
      setScalingData(data as ScalingPreviewData);
      const cols = Object.keys(data);
      if (cols.length > 0) setSelectedCol(cols[0]);
      setShowScalingModal(true);
    } catch {
      alert('Gagal mengambil preview scaling.');
    } finally {
      setLoadingScaling(false);
    }
  };

  const handleAskAi = async (rec: Recommendation) => {
    setLoadingAi(rec.id);
    try {
      const res = await explainRecommendation({
        category: rec.category,
        column_name: rec.column,
        recommendation: rec.recommended_action,
        score: rec.score,
        evidence: rec.evidence.reduce((acc: Record<string, string>, item: string) => {
          const delimiter = item.includes('=') ? '=' : item.includes(':') ? ':' : null;
          if (delimiter) {
            const idx = item.indexOf(delimiter);
            const key = item.slice(0, idx).trim();
            const val = item.slice(idx + 1).trim();
            if (key) acc[key] = val;
          } else if (item.trim()) {
            acc[item.trim()] = 'true';
          }
          return acc;
        }, {}),
        rationale: rec.rationale,
        alternative_actions: rec.alternatives,
      });
      setAiExplanation(res as Record<string, any>);
    } catch {
      setAiExplanation({ headline: 'AI Explanation', detailed_explanation: rec.rationale });
    } finally {
      setLoadingAi(null);
    }
  };

  useEffect(() => {
    if (!projectId) return;
    (async () => {
      const datasets = await listDatasets(projectId);
      if (!datasets.length) { setLoading(false); return; }
      const latest = datasets[datasets.length - 1];
      const [r, d] = await Promise.all([
        getRecommendations(latest.id),
        listDecisions(projectId),
      ]);
      setRecs(r);
      const decMap: Record<string, WorkflowDecision> = {};
      d.forEach(dec => {
        const key = `${dec.category}__${dec.column_name ?? 'dataset'}`;
        decMap[key] = dec;
      });
      setDecisions(decMap);
      setLoading(false);
    })();
  }, [projectId]);

  const decisionKey = (rec: Recommendation) =>
    `${rec.category}__${rec.column ?? 'dataset'}`;

  const getSelectedAction = (rec: Recommendation) => {
    const key = decisionKey(rec);
    if (overrides[key]) return overrides[key];
    if (decisions[key]) return decisions[key].selected_action;
    return null;
  };

  const getStatus = (rec: Recommendation): string => {
    const key = decisionKey(rec);
    if (decisions[key]) return decisions[key].status;
    return 'proposed';
  };

  const handleDecide = async (rec: Recommendation, action: string, status: string) => {
    if (!projectId) return;
    const key = decisionKey(rec);
    setSaving(s => ({ ...s, [key]: true }));
    try {
      const dec = await submitDecision(projectId, {
        recommendation_id: rec.id,
        category: rec.category,
        column_name: rec.column ?? undefined,
        selected_action: action,
        status,
      });
      setDecisions(d => ({ ...d, [key]: dec }));
      setOverrides(o => { const n = { ...o }; delete n[key]; return n; });
    } finally {
      setSaving(s => ({ ...s, [key]: false }));
    }
  };

  const acceptAll = async () => {
    for (const rec of recs) {
      const key = decisionKey(rec);
      if (!decisions[key]) {
        await handleDecide(rec, rec.recommended_action, 'accepted');
      }
    }
  };

  const categories = ['all', ...Array.from(new Set(recs.map(r => r.category)))];
  const filtered = filterCat === 'all' ? recs : recs.filter(r => r.category === filterCat);
  const acceptedCount = recs.filter(r => ['accepted', 'overridden'].includes(getStatus(r))).length;

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: '1rem' }}>
      <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      <div style={{ color: 'var(--text-secondary)' }}>Generating recommendations…</div>
    </div>
  );

  const activeColData = selectedCol && scalingData ? scalingData[selectedCol] : null;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Data Treatment</h1>
          <p className="page-subtitle">
            Review and approve preprocessing recommendations —&nbsp;
            <strong style={{ color: 'var(--success)' }}>{acceptedCount}</strong> / {recs.length} decided
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={handleFetchScalingPreview} disabled={loadingScaling}>
            <BarChart2 size={16} style={{ color: '#89b4fa' }} />
            {loadingScaling ? 'Loading Preview…' : 'Preview Scaling (Before vs After)'}
          </button>
          <button className="btn btn-secondary" onClick={acceptAll}>
            <CheckCircle2 size={16} /> Accept All Recommendations
          </button>
          <button
            className="btn btn-primary"
            onClick={() => navigate(`/projects/${projectId}/pipeline`)}
          >
            Build Pipeline <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
        {[
          { label: 'Total Issues', value: recs.length },
          { label: 'Accepted', value: recs.filter(r => getStatus(r) === 'accepted').length, color: 'var(--success)' },
          { label: 'Overridden', value: recs.filter(r => getStatus(r) === 'overridden').length, color: 'var(--warning)' },
          { label: 'Pending', value: recs.filter(r => getStatus(r) === 'proposed').length, color: 'var(--text-secondary)' },
        ].map(s => (
          <div key={s.label} className="metric-card">
            <div className="metric-label">{s.label}</div>
            <div className="metric-value" style={{ fontSize: '1.75rem', color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Category filter */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {categories.map(cat => (
          <button
            key={cat}
            className={`btn btn-sm ${filterCat === cat ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setFilterCat(cat)}
          >
            {cat === 'all' ? 'All Categories' : CATEGORY_LABELS[cat] ?? cat}
            <span className="badge badge-muted" style={{ marginLeft: '0.25rem' }}>
              {cat === 'all' ? recs.length : recs.filter(r => r.category === cat).length}
            </span>
          </button>
        ))}
      </div>

      {/* Decision Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {filtered.map(rec => {
          const key = decisionKey(rec);
          const status = getStatus(rec);
          const selectedAction = getSelectedAction(rec);
          const isSaving = saving[key];

          return (
            <div key={rec.id} className={`decision-card ${status !== 'proposed' ? status : ''}`}>
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.375rem' }}>
                    <span className={`badge ${CATEGORY_COLORS[rec.category] ?? 'badge-muted'}`}>
                      {CATEGORY_LABELS[rec.category] ?? rec.category}
                    </span>
                    {rec.column && (
                      <code style={{ fontSize: '0.85rem', color: 'var(--accent-hover)', background: 'var(--bg-elevated)', padding: '0.1rem 0.5rem', borderRadius: 4 }}>
                        {rec.column}
                      </code>
                    )}
                    {status === 'accepted' && <span className="badge badge-success"><CheckCircle2 size={10} /> Accepted</span>}
                    {status === 'overridden' && <span className="badge badge-warning"><RotateCcw size={10} /> Overridden</span>}
                    {status === 'rejected' && <span className="badge badge-danger"><X size={10} /> Rejected</span>}
                  </div>
                </div>

                {/* Score */}
                <div style={{ minWidth: 120 }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem', textAlign: 'right' }}>
                    Rec. Score
                  </div>
                  <ScoreBar score={rec.score} />
                </div>
              </div>

              {/* Evidence */}
              <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
                {rec.evidence.map(e => (
                  <code key={e} style={{
                    fontSize: '0.72rem', color: 'var(--text-secondary)',
                    background: 'var(--bg-elevated)', padding: '0.15rem 0.5rem', borderRadius: 4,
                    border: '1px solid var(--border)',
                  }}>
                    {e}
                  </code>
                ))}
              </div>

              {/* Rationale */}
              <div style={{
                background: 'var(--bg-elevated)', borderRadius: 8,
                padding: '0.75rem 1rem', fontSize: '0.875rem',
                color: 'var(--text-secondary)', borderLeft: '3px solid var(--border-highlight)',
              }}>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                  <Info size={14} style={{ flexShrink: 0, marginTop: '0.15rem', color: 'var(--accent)' }} />
                  {rec.rationale}
                </div>
              </div>

              {/* Action buttons */}
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 500 }}>
                  Recommended:&nbsp;
                  <strong style={{ color: 'var(--accent-hover)' }}>
                    {rec.recommended_action.replace(/_/g, ' ')}
                  </strong>
                  &nbsp;— or choose an alternative:
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                  {[rec.recommended_action, ...rec.alternatives].map(action => (
                    <ActionButton
                      key={action}
                      action={action}
                      selected={selectedAction === action}
                      onClick={() => setOverrides(o => ({ ...o, [key]: action }))}
                    />
                  ))}
                </div>
              </div>

              {/* Accept / Reject / Ask AI buttons */}
              <div style={{ display: 'flex', gap: '0.75rem', paddingTop: '0.25rem', borderTop: '1px solid var(--border)', flexWrap: 'wrap', alignItems: 'center' }}>
                <button
                  className="btn btn-success btn-sm"
                  disabled={isSaving}
                  onClick={() => handleDecide(
                    rec,
                    overrides[key] ?? rec.recommended_action,
                    overrides[key] && overrides[key] !== rec.recommended_action ? 'overridden' : 'accepted'
                  )}
                >
                  {isSaving ? <span className="spinner" /> : <CheckCircle2 size={14} />}
                  {selectedAction && selectedAction !== rec.recommended_action ? 'Apply Override' : 'Accept Recommendation'}
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={isSaving}
                  onClick={() => handleDecide(rec, rec.recommended_action, 'rejected')}
                >
                  <X size={14} /> Skip
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  disabled={loadingAi === rec.id}
                  onClick={() => handleAskAi(rec)}
                  style={{ marginLeft: 'auto', background: 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2))', border: '1px solid rgba(99,102,241,0.4)' }}
                >
                  <Sparkles size={13} style={{ color: '#cba6f7' }} />
                  {loadingAi === rec.id ? 'Thinking…' : 'Ask AI: Why this recommendation?'}
                </button>
              </div>
            </div>
          );
        })}
        {/* MODAL VISUALISASI SCALING (BEFORE VS AFTER) */}
        {showScalingModal && scalingData && createPortal(
          <div
            onClick={() => setShowScalingModal(false)}
            style={{
              position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 10000,
              display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1.5rem',
              backdropFilter: 'blur(4px)',
            }}
          >
            <div
              className="animate-slide-up"
              onClick={e => e.stopPropagation()}
              style={{
                background: '#181825', border: '1px solid #313244', borderRadius: '16px',
                maxWidth: '850px', width: '100%', padding: '1.75rem', color: '#cdd6f4', maxHeight: '90vh', overflowY: 'auto',
                boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid #313244', paddingBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <BarChart2 size={22} style={{ color: '#89b4fa' }} />
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
                    Visualisasi Dampak Scaling (Before vs After)
                  </h3>
                </div>
                <button onClick={() => setShowScalingModal(false)} style={{ background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer' }}>
                  <X size={20} />
                </button>
              </div>

              {/* Selector Kolom Numerik */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', background: '#1e1e2e', padding: '0.75rem 1rem', borderRadius: '10px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#a6adc8' }}>Pilih Kolom Numerik:</label>
                <select
                  value={selectedCol ?? ''}
                  onChange={e => setSelectedCol(e.target.value)}
                  style={{
                    background: '#313244', border: '1px solid #45475a', color: '#cdd6f4',
                    padding: '0.4rem 0.8rem', borderRadius: '6px', fontSize: '0.85rem', outline: 'none',
                  }}
                >
                  {Object.keys(scalingData).map(col => (
                    <option key={col} value={col}>{col} ({scalingData[col].scaling_action})</option>
                  ))}
                </select>
              </div>

              {activeColData ? (
                <div>
                  <div style={{ fontSize: '0.9rem', color: '#89b4fa', fontWeight: 600, marginBottom: '1rem' }}>
                    Aksi Transformasi: <span className="badge badge-primary">{activeColData.scaling_action.replace(/_/g, ' ')}</span>
                  </div>

                  <div className="grid-2" style={{ gap: '1.25rem' }}>
                    {/* Cards Before Scaling */}
                    <div style={{ background: '#1e1e2e', border: '1px solid #313244', borderRadius: '12px', padding: '1rem' }}>
                      <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#f38ba8', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span>Data Mentah (Before)</span>
                        <span style={{ fontSize: '0.75rem', background: '#313244', padding: '0.2rem 0.5rem', borderRadius: '4px', color: '#a6adc8' }}>Raw Scale</span>
                      </div>

                      <HistogramSvg
                        counts={activeColData.raw.counts}
                        color="#f38ba8"
                        minVal={activeColData.raw.min ?? undefined}
                        maxVal={activeColData.raw.max ?? undefined}
                      />

                      {/* Ringkasan Statistik Before */}
                      <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Mean:</span> <strong>{activeColData.raw.mean?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Std Dev:</span> <strong>{activeColData.raw.std?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Min / Max:</span> <strong>{activeColData.raw.min?.toFixed(1)} / {activeColData.raw.max?.toFixed(1)}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Skewness:</span> <strong>{activeColData.raw.skewness?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                      </div>
                    </div>

                    {/* Cards After Scaling */}
                    <div style={{ background: '#1e1e2e', border: '1px solid #313244', borderRadius: '12px', padding: '1rem' }}>
                      <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#a6e3a1', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span>Data Diskalakan (After)</span>
                        <span style={{ fontSize: '0.75rem', background: '#313244', padding: '0.2rem 0.5rem', borderRadius: '4px', color: '#a6adc8' }}>Transformed Scale</span>
                      </div>

                      <HistogramSvg
                        counts={activeColData.transformed.counts}
                        color="#a6e3a1"
                        minVal={activeColData.transformed.min ?? undefined}
                        maxVal={activeColData.transformed.max ?? undefined}
                      />

                      {/* Ringkasan Statistik After */}
                      <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Mean:</span> <strong>{activeColData.transformed.mean?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Std Dev:</span> <strong>{activeColData.transformed.std?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Min / Max:</span> <strong>{activeColData.transformed.min?.toFixed(2)} / {activeColData.transformed.max?.toFixed(2)}</strong>
                        </div>
                        <div style={{ background: '#181825', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                          <span style={{ color: '#a6adc8' }}>Skewness:</span> <strong>{activeColData.transformed.skewness?.toFixed(2) ?? 'N/A'}</strong>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: '1.25rem', background: '#181825', padding: '0.85rem 1.1rem', borderRadius: '10px', fontSize: '0.83rem', color: '#a6adc8', borderLeft: '4px solid #89b4fa', lineHeight: 1.6 }}>
                    💡 <strong>Mengapa Tinggi Grafik Bar Sama?</strong> <code>Standard Scaling</code> & <code>Min-Max Scaling</code> bersifat <em>linear transformation</em>. Transformasi ini mempertahankan proporsi data (sehingga tinggi bar frekuensi di tiap bin 100% sama), tetapi <strong>mereset dan mengubah rentang nilai sumbu X</strong> (perhatikan perubahan label Min/Max di bawah grafik: <code>0.0 ~ 8.0</code> berubah menjadi <code>-2.76 ~ 7.45</code>, serta <code>Mean → -0.00</code> dan <code>Std Dev → 1.00</code>).
                  </div>
                </div>
              ) : (
                <div style={{ padding: '2rem', textAlign: 'center', color: '#a6adc8' }}>Tidak ada data kolom numerik.</div>
              )}

              <div style={{ textAlign: 'right', marginTop: '1.5rem' }}>
                <button className="btn btn-primary" onClick={() => setShowScalingModal(false)}>
                  Tutup Visualisasi
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

        {/* AI Explanation Modal */}
        {aiExplanation && createPortal(
          <div
            onClick={() => setAiExplanation(null)}
            style={{
              position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 10000,
              display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem',
              backdropFilter: 'blur(4px)',
            }}
          >
            <div
              className="animate-slide-up"
              onClick={e => e.stopPropagation()}
              style={{
                background: '#181825', border: '1px solid #313244', borderRadius: '16px',
                maxWidth: '580px', width: '100%', padding: '1.75rem', color: '#cdd6f4',
                boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <Bot size={22} style={{ color: '#cba6f7' }} />
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0 }}>Penjelasan AI Assistant</h3>
                </div>
                <button onClick={() => setAiExplanation(null)} style={{ background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer' }}>
                  <X size={18} />
                </button>
              </div>

              <div style={{ background: '#1e1e2e', padding: '1.1rem', borderRadius: '12px', marginBottom: '1rem', borderLeft: '4px solid #6366f1' }}>
                <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#cba6f7', marginBottom: '0.4rem' }}>
                  {aiExplanation.headline || 'Analisis Preprocessing'}
                </div>
                <p style={{ fontSize: '0.875rem', color: '#cdd6f4', lineHeight: 1.6, margin: 0 }}>
                  {aiExplanation.detailed_explanation || 'Penjelasan tidak tersedia.'}
                </p>
              </div>

              {Array.isArray(aiExplanation.key_evidence) && aiExplanation.key_evidence.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.78rem', color: '#a6adc8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem', fontWeight: 600 }}>
                    Bukti Statistik Kunci (Evidence)
                  </div>
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    {aiExplanation.key_evidence.map((ev: string, idx: number) => (
                      <span key={idx} style={{ background: '#313244', padding: '0.25rem 0.65rem', borderRadius: '6px', fontSize: '0.78rem', border: '1px solid #45475a' }}>
                        {ev}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {aiExplanation.alternative_rationale && (
                <div style={{ background: '#181825', border: '1px dashed #313244', padding: '0.75rem 1rem', borderRadius: '8px', fontSize: '0.8rem', color: '#a6adc8', marginBottom: '1rem' }}>
                  <strong style={{ color: '#89b4fa' }}>Pertimbangan Alternatif:</strong>
                  <div style={{ marginTop: '0.25rem', whiteSpace: 'pre-line' }}>{aiExplanation.alternative_rationale}</div>
                </div>
              )}

              <div style={{ textAlign: 'right', marginTop: '1.25rem' }}>
                <button className="btn btn-primary" onClick={() => setAiExplanation(null)}>
                  Paham
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

        {filtered.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">✅</div>
            <h3>No issues found</h3>
            <p>Dataset looks clean for this category.</p>
          </div>
        )}
      </div>
    </div>
  );
}
