import { useEffect, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useParams } from 'react-router-dom';
import {
  Activity, CheckCircle2, AlertTriangle, XCircle, RefreshCw,
  Sliders, Trash2, Play, BarChart2, X, Layers, Check,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import { getProjectDrift, simulateDriftTraffic, clearDriftLogs } from '../api/client';
import type { DriftReportResponse, FeatureDriftDetail } from '../types';

export function DriftMonitoringPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [report, setReport] = useState<DriftReportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<FeatureDriftDetail | null>(null);

  const fetchDrift = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getProjectDrift(projectId);
      setReport(data);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (err as Error)?.message ?? 'Gagal memuat laporan data drift.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchDrift();
  }, [fetchDrift]);

  const handleSimulate = async (driftType: 'normal' | 'drifted') => {
    if (!projectId) return;
    setActionLoading(driftType);
    setActionMessage(null);
    try {
      const res = await simulateDriftTraffic(projectId, driftType, 30);
      setActionMessage(res.message);
      await fetchDrift();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (err as Error)?.message ?? 'Gagal mensimulasikan trafik.';
      setError(msg);
    } finally {
      setActionLoading(null);
    }
  };

  const handleClearLogs = async () => {
    if (!projectId || !window.confirm('Hapus seluruh log inferensi untuk mereset data monitoring?')) return;
    setActionLoading('clear');
    setActionMessage(null);
    try {
      const res = await clearDriftLogs(projectId);
      setActionMessage(res.message);
      await fetchDrift();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (err as Error)?.message ?? 'Gagal menghapus log.';
      setError(msg);
    } finally {
      setActionLoading(null);
    }
  };

  // ── Status Banner Configuration ──
  const renderStatusBanner = () => {
    if (!report) return null;

    if (report.status === 'INSUFFICIENT_DATA') {
      return (
        <div className="card" style={{
          padding: '1.5rem',
          marginBottom: '1.5rem',
          borderLeft: '4px solid #6366f1',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
        }}>
          <div style={{
            width: 46, height: 46, borderRadius: 12,
            background: 'rgba(99,102,241,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={24} color="#6366f1" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
              Awaiting Production Traffic
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              {report.message || 'Log inferensi belum mencukupi untuk uji statistik. Lakukan live prediction atau klik tombol simulasi di bawah.'}
            </div>
          </div>
        </div>
      );
    }

    if (report.status === 'DRIFT_DETECTED') {
      return (
        <div className="card" style={{
          padding: '1.5rem',
          marginBottom: '1.5rem',
          borderLeft: '4px solid #ef4444',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
        }}>
          <div style={{
            width: 46, height: 46, borderRadius: 12,
            background: 'rgba(239,68,68,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <XCircle size={24} color="#ef4444" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#b91c1c' }}>
              Significant Data Drift Detected
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Sebanyak {report.drifted_features_count} dari {report.monitored_features_count} fitur ({((report.drift_share || 0) * 100).toFixed(0)}%) mengalami pergeseran distribusi yang signifikan (p &lt; 0.05 atau PSI &ge; 0.25). Disarankan melakukan retraining model.
            </div>
          </div>
        </div>
      );
    }

    if (report.status === 'WARNING') {
      return (
        <div className="card" style={{
          padding: '1.5rem',
          marginBottom: '1.5rem',
          borderLeft: '4px solid #f59e0b',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
        }}>
          <div style={{
            width: 46, height: 46, borderRadius: 12,
            background: 'rgba(245,158,11,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <AlertTriangle size={24} color="#f59e0b" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#b45309' }}>
              Moderate Drift Warning
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Terdapat {report.drifted_features_count} fitur yang mengalami pergeseran statistik. Pantau data inference selanjutnya untuk memastikan akurasi model tetap terjaga.
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="card" style={{
        padding: '1.5rem',
        marginBottom: '1.5rem',
        borderLeft: '4px solid #10b981',
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        flexWrap: 'wrap',
      }}>
        <div style={{
          width: 46, height: 46, borderRadius: 12,
          background: 'rgba(16,185,129,0.12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <CheckCircle2 size={24} color="#10b981" />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#047857' }}>
            Model Distribution Healthy (No Drift)
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Seluruh fitur produksi berada dalam rentang variasi normal terhadap dataset training. Model beroperasi dengan stabil.
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '2rem' }} className="animate-fade-in">
      {/* Header */}
      <div className="page-header" style={{ marginBottom: '1.5rem' }}>
        <Activity size={22} color="#4338ca" />
        <div>
          <h1 className="page-title">Model &amp; Data Drift Monitor</h1>
          <p className="page-subtitle">
            Memantau stabilitas distribusi fitur data produksi vs data training menggunakan uji statistik KS-Test dan PSI.
          </p>
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: '1.5rem' }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {actionMessage && (
        <div className="alert alert-success" style={{ marginBottom: '1.5rem' }}>
          <Check size={16} />
          {actionMessage}
        </div>
      )}

      {/* Main Status Banner */}
      {renderStatusBanner()}

      {/* Metrics Summary Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1.25rem',
        marginBottom: '1.75rem',
      }}>
        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Total Inferences Logged
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
            {report?.total_predictions_logged ?? 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Sampel produksi terkumpul
          </div>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Monitored Features
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#4338ca', marginTop: '0.35rem' }}>
            {report?.monitored_features_count ?? 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Fitur dalam pipeline model
          </div>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Drifted Features
          </div>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 700,
            color: (report?.drifted_features_count ?? 0) > 0 ? '#ef4444' : '#10b981',
            marginTop: '0.35rem',
          }}>
            {report?.drifted_features_count ?? 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Kolom dengan pergeseran
          </div>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Drift Share
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
            {report ? `${((report.drift_share || 0) * 100).toFixed(1)}%` : '0%'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Rasio fitur terpengaruh
          </div>
        </div>
      </div>

      {/* Control Action Toolbar */}
      <div className="card" style={{
        padding: '1.25rem',
        marginBottom: '1.75rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
            Traffic &amp; Simulation Controls
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Simulasikan masukan data produksi untuk menguji deteksi anomali dan pergeseran distribusi data secara instan.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
          <button
            className="btn btn-secondary"
            onClick={() => handleSimulate('normal')}
            disabled={actionLoading !== null}
            style={{ fontSize: '0.82rem', padding: '0.5rem 0.85rem' }}
          >
            {actionLoading === 'normal' ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
            Simulate Normal Traffic (+30)
          </button>

          <button
            className="btn btn-primary"
            onClick={() => handleSimulate('drifted')}
            disabled={actionLoading !== null}
            style={{ fontSize: '0.82rem', padding: '0.5rem 0.85rem' }}
          >
            {actionLoading === 'drifted' ? <RefreshCw size={14} className="animate-spin" /> : <Sliders size={14} />}
            Simulate Drifted Traffic (+30)
          </button>

          <button
            className="btn btn-outline"
            onClick={fetchDrift}
            disabled={loading}
            style={{ fontSize: '0.82rem', padding: '0.5rem 0.85rem' }}
            title="Refresh Analisis"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>

          <button
            className="btn btn-outline"
            onClick={handleClearLogs}
            disabled={actionLoading !== null || (report?.total_predictions_logged ?? 0) === 0}
            style={{ fontSize: '0.82rem', padding: '0.5rem 0.85rem', color: '#b91c1c' }}
            title="Bersihkan seluruh log prediksi"
          >
            <Trash2 size={14} />
            Clear Logs
          </button>
        </div>
      </div>

      {/* Feature Drift Breakdown Table */}
      <div className="card" style={{ padding: '1.5rem', overflowX: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Feature Drift Analysis Breakdown
            </h2>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Pengujian statistik per-kolom: KS-Test untuk numerik (p &lt; 0.05) dan PSI untuk kategorikal (PSI &ge; 0.25).
            </p>
          </div>
        </div>

        {(!report?.feature_details || report.feature_details.length === 0) ? (
          <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={36} style={{ opacity: 0.4, marginBottom: '0.75rem' }} />
            <p style={{ fontSize: '0.9rem' }}>Belum ada data analisis fitur yang tersedia.</p>
            <p style={{ fontSize: '0.78rem', marginTop: '0.25rem' }}>
              Jalankan simulasi trafik atau kirim live prediction untuk memunculkan tabel analisis.
            </p>
          </div>
        ) : (
          <table className="table" style={{ width: '100%', fontSize: '0.85rem' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Feature</th>
                <th style={{ textAlign: 'left' }}>Type</th>
                <th style={{ textAlign: 'left' }}>Test Method</th>
                <th style={{ textAlign: 'center' }}>Test Statistic</th>
                <th style={{ textAlign: 'center' }}>P-Value / Metric</th>
                <th style={{ textAlign: 'left' }}>Baseline vs Current</th>
                <th style={{ textAlign: 'center' }}>Status</th>
                <th style={{ textAlign: 'center' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {report.feature_details.map((feat) => {
                const isDrift = feat.drift_detected;
                return (
                  <tr key={feat.feature_name}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {feat.feature_name}
                    </td>
                    <td>
                      <span className="badge" style={{ fontSize: '0.72rem', textTransform: 'capitalize' }}>
                        {feat.dtype}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                      {feat.test_name}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'monospace', fontWeight: 600 }}>
                      {feat.statistic.toFixed(4)}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'monospace' }}>
                      {feat.p_value !== null && feat.p_value !== undefined
                        ? `p = ${feat.p_value < 0.0001 ? '< 0.0001' : feat.p_value.toFixed(4)}`
                        : `PSI = ${feat.statistic.toFixed(3)}`}
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      {feat.dtype === 'numeric' ? (
                        <span>
                          μ: {feat.baseline_mean ?? '—'} → {feat.current_mean ?? '—'}
                          {feat.wasserstein_distance ? ` (dist: ${feat.wasserstein_distance})` : ''}
                        </span>
                      ) : (
                        <span>Kategori terpantau</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {isDrift ? (
                        <span className="badge badge-error" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                          <XCircle size={12} style={{ marginRight: 4 }} /> Drift
                        </span>
                      ) : feat.status === 'MODERATE' ? (
                        <span className="badge badge-warning" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                          <AlertTriangle size={12} style={{ marginRight: 4 }} /> Moderate
                        </span>
                      ) : (
                        <span className="badge badge-success" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                          <CheckCircle2 size={12} style={{ marginRight: 4 }} /> Stable
                        </span>
                      )}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}
                        onClick={() => setSelectedFeature(feat)}
                      >
                        <BarChart2 size={12} /> Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Distribution Comparison Modal (via Portal) */}
      {selectedFeature && createPortal(
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(15, 23, 42, 0.45)',
          backdropFilter: 'blur(3px)',
          zIndex: 9999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem',
        }}>
          <div className="card animate-fade-in" style={{
            maxWidth: 720,
            width: '100%',
            background: 'var(--bg-card)',
            boxShadow: 'var(--neu-raised)',
            borderRadius: '16px',
            padding: '1.75rem',
            position: 'relative',
          }}>
            {/* Modal Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <div style={{
                  width: 36, height: 36, borderRadius: 10,
                  background: 'rgba(67,56,202,0.12)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <BarChart2 size={18} color="#4338ca" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    Distribution Shift: {selectedFeature.feature_name}
                  </h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Perbandingan persentase frekuensi antara Baseline (Training) vs Current (Production)
                  </p>
                </div>
              </div>
              <button
                className="btn btn-icon"
                onClick={() => setSelectedFeature(null)}
                aria-label="Tutup"
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Stat Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '0.75rem',
              marginBottom: '1.25rem',
            }}>
              <div style={{
                background: 'var(--bg-inset)',
                padding: '0.75rem',
                borderRadius: 10,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Test Method</div>
                <div style={{ fontWeight: 700, fontSize: '0.85rem', marginTop: 2, color: 'var(--text-primary)' }}>
                  {selectedFeature.test_name}
                </div>
              </div>

              <div style={{
                background: 'var(--bg-inset)',
                padding: '0.75rem',
                borderRadius: 10,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Test Statistic</div>
                <div style={{ fontWeight: 700, fontSize: '0.85rem', marginTop: 2, color: 'var(--text-primary)' }}>
                  {selectedFeature.statistic.toFixed(4)}
                </div>
              </div>

              <div style={{
                background: 'var(--bg-inset)',
                padding: '0.75rem',
                borderRadius: 10,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Drift Status</div>
                <div style={{
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  marginTop: 2,
                  color: selectedFeature.drift_detected ? '#ef4444' : '#10b981',
                }}>
                  {selectedFeature.drift_detected ? 'DRIFT DETECTED' : 'STABLE'}
                </div>
              </div>
            </div>

            {/* Recharts Dual Distribution BarChart */}
            <div style={{ width: '100%', height: 280, marginTop: '0.5rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={selectedFeature.distribution_chart} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="bin"
                    tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                    angle={-20}
                    textAnchor="end"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                    unit="%"
                  />
                  <Tooltip
                    formatter={(value: any) => [`${value}%`]}
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Legend
                    verticalAlign="top"
                    wrapperStyle={{ paddingBottom: 10, fontSize: 12 }}
                  />
                  <Bar
                    dataKey="baseline_pct"
                    name="Baseline (Training)"
                    fill="#6366f1"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="current_pct"
                    name="Current (Production)"
                    fill={selectedFeature.drift_detected ? '#ef4444' : '#10b981'}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Modal Footer */}
            <div style={{ marginTop: '1.25rem', textAlign: 'right' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setSelectedFeature(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
