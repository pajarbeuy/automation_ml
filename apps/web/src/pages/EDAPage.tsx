import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie
} from 'recharts';
import { listDatasets, getProfile, getEDA, getProject } from '../api/client';
import type { DatasetProfile, DatasetEDA, Project } from '../types';
import { AlertTriangle, CheckCircle, ChevronRight, Database, Info } from 'lucide-react';

type Tab = 'overview' | 'missing' | 'distributions' | 'correlations' | 'target';

const COLORS = ['#6366f1', '#a855f7', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'];

function formatCompactNumber(num: number): string {
  if (Math.abs(num) >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
  if (Math.abs(num) >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (Math.abs(num) >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
  return Number.isInteger(num) ? num.toString() : num.toFixed(2);
}

export function EDAPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>('overview');
  const [project, setProject] = useState<Project | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [eda, setEda] = useState<DatasetEDA | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!projectId) return;
    (async () => {
      try {
        const [datasets, proj] = await Promise.all([
          listDatasets(projectId),
          getProject(projectId).catch(() => null),
        ]);
        setProject(proj);
        if (!datasets.length) { setLoading(false); return; }
        const latest = datasets[datasets.length - 1];
        const [p, e] = await Promise.all([getProfile(latest.id), getEDA(latest.id)]);
        setProfile(p);
        setEda(e);
      } finally {
        setLoading(false);
      }
    })();
  }, [projectId]);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: '1rem' }}>
      <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      <div style={{ color: 'var(--text-secondary)' }}>Loading EDA workspace…</div>
    </div>
  );

  if (!profile?.profile_json) return (
    <div className="empty-state">
      <div className="empty-state-icon"><Database size={48} strokeWidth={1} /></div>
      <h3>No dataset found</h3>
      <p style={{ marginBottom: '1.5rem' }}>Please upload a dataset first.</p>
      <button className="btn btn-primary" onClick={() => navigate(`/projects/${projectId}/upload`)}>
        Go to Upload
      </button>
    </div>
  );

  const pj = profile.profile_json;
  const edaData = eda?.eda_json;
  const features = pj.features;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">EDA Workspace</h1>
          <p className="page-subtitle">{profile.filename} — {pj.rows.toLocaleString()} rows × {pj.columns} cols</p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => navigate(`/projects/${projectId}/treatment`)}
        >
          Review Recommendations <ChevronRight size={16} />
        </button>
      </div>

      {/* Stats row */}
      <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
        {[
          { label: 'Rows', value: pj.rows.toLocaleString() },
          { label: 'Columns', value: pj.columns },
          { label: 'Duplicate Rows', value: pj.duplicate_rows, danger: pj.duplicate_rows > 0 },
          { label: 'Target', value: pj.target },
        ].map(s => (
          <div key={s.label} className="metric-card">
            <div className="metric-label">{s.label}</div>
            <div className="metric-value" style={{ fontSize: '1.4rem', color: s.danger ? 'var(--warning)' : undefined }}>
              {String(s.value)}
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="tabs">
        {(['overview', 'missing', 'distributions', 'correlations', 'target'] as Tab[]).map(t => (
          <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ marginBottom: '1rem' }}>Feature Summary</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Unique</th>
                    <th>Constant</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {features.map(f => (
                    <tr key={f.name}>
                      <td style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 500 }}>{f.name}</td>
                      <td><span className={`badge ${f.dtype === 'numeric' ? 'badge-info' : f.dtype === 'categorical' ? 'badge-primary' : 'badge-muted'}`}>{f.dtype}</span></td>
                      <td>
                        {f.missing_count > 0 ? (
                          <span style={{ color: 'var(--warning)' }}>
                            {f.missing_count} ({(f.missing_rate * 100).toFixed(1)}%)
                          </span>
                        ) : <span style={{ color: 'var(--success)' }}>None</span>}
                      </td>
                      <td style={{ color: 'var(--text-secondary)' }}>{f.unique_count.toLocaleString()}</td>
                      <td>
                        {f.constant
                          ? <span className="badge badge-warning" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}><AlertTriangle size={11} /> Constant</span>
                          : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                      </td>
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {f.high_cardinality && <span className="badge badge-warning">High cardinality</span>}
                        {f.outlier_rate_iqr && f.outlier_rate_iqr > 0.05 && (
                          <span className="badge badge-danger" style={{ marginLeft: '0.25rem' }}>Outliers</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Missing Values Tab */}
      {tab === 'missing' && (
        <div>
          {edaData?.missing_value_summary.length === 0 ? (
            <div className="alert alert-success">
              <CheckCircle size={16} />
              No missing values detected in this dataset!
            </div>
          ) : (
            <div className="card">
              <h3 style={{ marginBottom: '1.25rem' }}>Missing Values by Column</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={edaData?.missing_value_summary} layout="vertical" margin={{ left: 80 }}>
                  <XAxis type="number" tickFormatter={v => `${v}%`} stroke="var(--text-muted)" />
                  <YAxis type="category" dataKey="column" stroke="var(--text-muted)" fontSize={12} />
                  <Tooltip
                    formatter={(v: any) => [`${v}%`, 'Missing Rate']}
                    contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
                  />
                  <Bar dataKey="missing_rate" fill="var(--warning)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Distributions Tab */}
      {tab === 'distributions' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {edaData && Object.entries(edaData.numerical_distributions).map(([col, dist]) => {
            const data = dist.bins.slice(0, -1).map((b, i) => ({
              bin: b.toFixed(1),
              count: dist.counts[i],
            }));
            return (
              <div key={col} className="card">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <h4>{col}</h4>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {dist.skewness !== null && Math.abs(dist.skewness) > 1 && (
                      <span className="badge badge-warning">Skewed ({dist.skewness.toFixed(2)})</span>
                    )}
                    {dist.outlier_rate_iqr > 5 && (
                      <span className="badge badge-danger">Outliers ({dist.outlier_rate_iqr.toFixed(1)}%)</span>
                    )}
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data}>
                    <XAxis dataKey="bin" stroke="var(--text-muted)" fontSize={11} />
                    <YAxis stroke="var(--text-muted)" fontSize={11} />
                    <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }} />
                    <Bar dataKey="count" fill="var(--accent)" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span>Mean: {dist.mean?.toFixed(2)}</span>
                  <span>Median: {dist.median?.toFixed(2)}</span>
                  <span>Std: {dist.std?.toFixed(2)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Correlations Tab */}
      {tab === 'correlations' && (
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Correlation Matrix (Pearson)</h3>
          {edaData?.correlation_matrix.columns.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', padding: '2rem', textAlign: 'center' }}>
              Not enough numeric features for correlation analysis.
            </div>
          ) : (
            <div className="table-wrap">
              <table style={{ fontSize: '0.8rem' }}>
                <thead>
                  <tr>
                    <th></th>
                    {edaData?.correlation_matrix.columns.map(c => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {edaData?.correlation_matrix.columns.map((row, i) => (
                    <tr key={row}>
                      <td style={{ fontWeight: 600 }}>{row}</td>
                      {edaData.correlation_matrix.matrix[i].map((v, j) => {
                        const val = v ?? 0;
                        const abs = Math.abs(val);
                        const bg = val > 0
                          ? `rgba(99,102,241,${abs * 0.7})`
                          : `rgba(239,68,68,${abs * 0.7})`;
                        return (
                          <td key={j} style={{ background: bg, textAlign: 'center', borderRadius: 4, fontFamily: 'monospace' }}>
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Target Tab */}
      {tab === 'target' && (() => {
        const rawDistEntries = Object.entries(pj.class_distribution || {});
        const uniqueTargetCount = rawDistEntries.length;
        const isNumericTarget =
          rawDistEntries.length > 0 &&
          rawDistEntries.every(([val]) => !isNaN(Number(val)) && val.trim() !== '');
        const isRegression =
          project?.task_type === 'regression' || (uniqueTargetCount > 10 && isNumericTarget);
        const isHighCardinality = !isRegression && uniqueTargetCount > 10;

        // Continuous / Regression Histogram preparation
        const numericValues = rawDistEntries
          .map(([v, p]) => ({ val: Number(v), prop: p }))
          .sort((a, b) => a.val - b.val);

        let targetHistogram: Array<{ range: string; proportion: number; count: number }> = [];
        let targetStats = { min: 0, max: 0, mean: 0, range: 0 };

        if (numericValues.length > 0) {
          const minVal = numericValues[0].val;
          const maxVal = numericValues[numericValues.length - 1].val;
          const meanVal = numericValues.reduce((acc, x) => acc + x.val * x.prop, 0);
          targetStats = {
            min: minVal,
            max: maxVal,
            mean: meanVal,
            range: maxVal - minVal,
          };

          const binCount = Math.min(10, Math.max(5, Math.ceil(Math.sqrt(numericValues.length))));
          const span = maxVal - minVal;
          const binStep = span > 0 ? span / binCount : 1;

          const bins: Array<{ min: number; max: number; label: string; proportion: number; count: number }> = [];
          for (let i = 0; i < binCount; i++) {
            const bStart = minVal + i * binStep;
            const bEnd = i === binCount - 1 ? maxVal : minVal + (i + 1) * binStep;
            bins.push({
              min: bStart,
              max: bEnd,
              label: `${formatCompactNumber(bStart)} - ${formatCompactNumber(bEnd)}`,
              proportion: 0,
              count: 0,
            });
          }

          for (const item of numericValues) {
            let assigned = false;
            for (let i = 0; i < bins.length; i++) {
              if (
                (i === bins.length - 1 && item.val >= bins[i].min && item.val <= bins[i].max) ||
                (item.val >= bins[i].min && item.val < bins[i].max)
              ) {
                bins[i].proportion += item.prop * 100;
                bins[i].count += 1;
                assigned = true;
                break;
              }
            }
            if (!assigned && bins.length > 0) {
              bins[bins.length - 1].proportion += item.prop * 100;
              bins[bins.length - 1].count += 1;
            }
          }

          targetHistogram = bins.map(b => ({
            range: b.label,
            count: b.count,
            proportion: Number(b.proportion.toFixed(2)),
          }));
        }

        const sortedDist = [...rawDistEntries].sort((a, b) => b[1] - a[1]);
        const top10 = sortedDist.slice(0, 10);

        if (isRegression) {
          return (
            <div className="grid-2">
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0 }}>Target Distribution (Histogram)</h3>
                  <span className="badge" style={{ background: 'var(--primary-subtle)', color: 'var(--primary)', fontWeight: 600, fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: 999 }}>
                    Regression Target
                  </span>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                  Distribusi data kontinu target <strong>{pj.target}</strong> terbagi ke dalam {targetHistogram.length} bin rentang nilai.
                </p>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={targetHistogram} margin={{ top: 10, right: 15, left: -10, bottom: 25 }}>
                    <XAxis
                      dataKey="range"
                      angle={-20}
                      textAnchor="end"
                      interval={0}
                      tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                      unit="%"
                    />
                    <Tooltip
                      formatter={(v: any) => [`${v}%`, 'Proporsi']}
                      contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
                    />
                    <Bar dataKey="proportion" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>

                <div className="alert alert-info" style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.85rem' }}>
                  <Info size={16} style={{ flexShrink: 0 }} />
                  <span>
                    Kolom target bersifat numerik kontinu dengan <strong>{uniqueTargetCount.toLocaleString()}</strong> nilai unik. Model regresi akan memprediksi nilai kontinu (tidak memerlukan class balancing).
                  </span>
                </div>
              </div>

              <div className="card">
                <h3 style={{ marginBottom: '1rem' }}>Target Summary & Statistics</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Target Variable</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{pj.target}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Task Type</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--primary)' }}>Regression</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Unique Numerical Values</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{uniqueTargetCount.toLocaleString()}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Minimum Value</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{targetStats.min.toLocaleString()} ({formatCompactNumber(targetStats.min)})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Maximum Value</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{targetStats.max.toLocaleString()} ({formatCompactNumber(targetStats.max)})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Approx. Mean</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{targetStats.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({formatCompactNumber(targetStats.mean)})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0.8rem', background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Range Spread</span>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{targetStats.range.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        }

        if (isHighCardinality) {
          return (
            <div className="grid-2">
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0 }}>Class Distribution (Top 10)</h3>
                  <span className="badge" style={{ background: 'var(--primary-subtle)', color: 'var(--primary)', fontWeight: 600, fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: 999 }}>
                    High Cardinality ({uniqueTargetCount} classes)
                  </span>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                  Menampilkan 10 kelas teratas untuk menjaga keterbacaan grafik.
                </p>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={top10.map(([cls, prop]) => ({ class: cls, proportion: Number((prop * 100).toFixed(2)) }))}
                    margin={{ top: 10, right: 15, left: -10, bottom: 25 }}
                  >
                    <XAxis
                      dataKey="class"
                      angle={-20}
                      textAnchor="end"
                      interval={0}
                      tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                      unit="%"
                    />
                    <Tooltip
                      formatter={(v: any) => [`${v}%`, 'Proporsi']}
                      contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
                    />
                    <Bar dataKey="proportion" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0 }}>Top Class Proportions</h3>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Top 10 dari {uniqueTargetCount}</span>
                </div>
                <div style={{ maxHeight: 310, overflowY: 'auto', paddingRight: '0.25rem' }}>
                  {top10.map(([cls, prop], i) => (
                    <div key={cls} style={{ marginBottom: '0.875rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.875rem' }}>
                        <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                          {cls}
                        </span>
                        <span style={{ color: 'var(--text-secondary)' }}>{(prop * 100).toFixed(2)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${Math.min(100, prop * 100)}%`, background: COLORS[i % COLORS.length] }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        }

        // Standard discrete classification (<= 10 classes)
        const dist = pj.class_distribution;
        const vals = Object.values(dist);
        const ratio = vals.length >= 2 ? Math.max(...vals) / Math.min(...vals) : 1;

        return (
          <div className="grid-2">
            <div className="card">
              <h3 style={{ marginBottom: '1rem' }}>Class Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={edaData?.class_distribution}
                    dataKey="proportion"
                    nameKey="class"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={uniqueTargetCount <= 5 ? (entry: any) => `${entry.class}: ${Number(entry.proportion).toFixed(1)}%` : undefined}
                  >
                    {edaData?.class_distribution?.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v: any) => `${Number(v).toFixed(1)}%`}
                    contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
                  />
                </PieChart>
              </ResponsiveContainer>

              {/* Imbalance warning only for true discrete classification */}
              {ratio > 3 && (
                <div className="alert alert-warning" style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle size={16} style={{ flexShrink: 0 }} />
                  <span>
                    Class imbalance detected ({ratio.toFixed(1)}:1 ratio). Consider using class_weight='balanced'.
                  </span>
                </div>
              )}
            </div>

            <div className="card">
              <h3 style={{ marginBottom: '1rem' }}>Class Proportions</h3>
              {Object.entries(pj.class_distribution).map(([cls, prop], i) => (
                <div key={cls} style={{ marginBottom: '0.875rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.875rem' }}>
                    <span style={{ fontWeight: 500 }}>{cls}</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{(prop * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${prop * 100}%`, background: COLORS[i % COLORS.length] }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
