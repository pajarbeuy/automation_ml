import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Brain, FolderOpen, Trash2, ChevronRight, Zap, Clock, Target, TrendingUp, CheckCircle2 } from 'lucide-react';
import { listProjects, createProject, deleteProject } from '../api/client';
import type { Project } from '../types';

function statusColor(status: string): string {
  switch (status) {
    case 'completed': return 'badge-success';
    case 'running': return 'badge-info';
    case 'failed': return 'badge-danger';
    case 'profiled': return 'badge-primary';
    default: return 'badge-muted';
  }
}

export function DashboardPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({ name: '', description: '', taskType: 'classification' });
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .finally(() => setLoading(false));
  }, []);

  const handleCreate = async () => {
    if (!form.name.trim()) return;
    setCreating(true);
    try {
      const project = await createProject({
        name: form.name,
        description: form.description,
        task_type: form.taskType,
      });
      navigate(`/projects/${project.id}/upload`);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm('Delete this project and all its data?')) return;
    await deleteProject(id);
    setProjects(p => p.filter(x => x.id !== id));
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>
      {/* Header */}
      <div style={{
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-surface)',
        padding: '1.25rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
          <div style={{
            width: 42, height: 42,
            background: 'linear-gradient(135deg, #6366f1, #a855f7)',
            borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Brain size={22} color="white" />
          </div>
          <div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>AutoML Human-in-the-Loop</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              From raw CSV to validated ML pipeline (Classification & Regression)
            </div>
          </div>
        </div>
        <button
          id="new-project-btn"
          className="btn btn-primary"
          onClick={() => setShowModal(true)}
        >
          <Plus size={16} /> New Project
        </button>
      </div>

      {/* Hero banner */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 50%, transparent 100%)',
        borderBottom: '1px solid var(--border)',
        padding: '3rem 2rem',
      }}>
        <div className="container">
          <h1 style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>
            Welcome to <span className="gradient-text">AutoML HIL</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', maxWidth: 580 }}>
            Build reproducible tabular ML pipelines with human-controlled preprocessing decisions.
            Supports both <strong>Classification</strong> and <strong>Regression</strong> tasks with automated feature engineering and hyperparameter tuning.
          </p>

          <div style={{ display: 'flex', gap: '2rem', marginTop: '2rem', flexWrap: 'wrap' }}>
            {[
              { icon: <Zap size={18} />, title: 'Automated Profiling', desc: 'Instant dataset analysis & EDA' },
              { icon: <Brain size={18} />, title: 'Smart Recommendations', desc: 'Evidence-based preprocessing advice' },
              { icon: <ChevronRight size={18} />, title: 'Advanced AutoML', desc: 'Feature engineering, tuning, & regression' },
            ].map((f) => (
              <div key={f.title} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                <div style={{ color: 'var(--accent)', marginTop: '0.1rem' }}>{f.icon}</div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{f.title}</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{f.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Projects grid */}
      <div className="container" style={{ padding: '2rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Projects</h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {projects.length} project{projects.length !== 1 ? 's' : ''}
          </span>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>
            <div className="spinner" style={{ margin: '0 auto 1rem' }} />
            Loading projects…
          </div>
        ) : projects.length === 0 ? (
          <div className="empty-state animate-fade-in">
            <div className="empty-state-icon"><FolderOpen size={48} strokeWidth={1} /></div>
            <h3>No projects yet</h3>
            <p style={{ marginBottom: '1.5rem' }}>Create your first project to get started</p>
            <button className="btn btn-primary" onClick={() => setShowModal(true)}>
              <Plus size={16} /> Create Project
            </button>
          </div>
        ) : (
          <div className="grid-3 animate-fade-in">
            {projects.map((p) => (
              <div
                key={p.id}
                className="card"
                style={{ cursor: 'pointer', position: 'relative', transition: 'all 0.2s' }}
                onClick={() => navigate(`/projects/${p.id}/upload`)}
                id={`project-card-${p.id}`}
              >
                <button
                  className="btn btn-ghost btn-icon"
                  style={{ position: 'absolute', top: '0.75rem', right: '0.75rem' }}
                  onClick={(e) => handleDelete(e, p.id)}
                  title="Delete project"
                >
                  <Trash2 size={14} />
                </button>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.875rem' }}>
                  <div style={{
                    width: 36, height: 36,
                    background: p.task_type === 'regression'
                      ? 'linear-gradient(135deg, rgba(16,185,129,0.3), rgba(59,130,246,0.3))'
                      : 'linear-gradient(135deg, rgba(99,102,241,0.3), rgba(168,85,247,0.3))',
                    borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    border: '1px solid rgba(99,102,241,0.3)',
                  }}>
                    {p.task_type === 'regression' ? (
                      <TrendingUp size={16} color="#10b981" />
                    ) : (
                      <Target size={16} color="var(--accent-hover)" />
                    )}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{p.name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                      {p.task_type || 'classification'}
                    </div>
                  </div>
                </div>

                {p.description && (
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.875rem' }}>
                    {p.description}
                  </p>
                )}

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.875rem' }}>
                  <span className={`badge ${statusColor(p.status)}`}>{p.status}</span>
                  {p.target_column && (
                    <span className="badge badge-muted">target: {p.target_column}</span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  <Clock size={12} />
                  {new Date(p.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create modal */}
      {showModal && (
        <div
          style={{
            position: 'fixed', inset: 0,
            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          }}
          onClick={() => setShowModal(false)}
        >
          <div
            className="card animate-fade-in"
            style={{ width: 500, maxWidth: '90vw' }}
            onClick={e => e.stopPropagation()}
          >
            <h3 style={{ marginBottom: '1.25rem' }}>New Project</h3>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label className="label">Project Name *</label>
              <input
                id="project-name-input"
                className="input"
                placeholder="e.g. Housing Price Regressor or Customer Churn"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                autoFocus
              />
            </div>

            <div className="form-group" style={{ marginBottom: '1.25rem' }}>
              <label className="label">Task Type (Jenis Tugas ML) *</label>
              <div className="grid-2" style={{ gap: '0.75rem' }}>
                {/* Classification Radio Card */}
                <div
                  onClick={() => setForm(f => ({ ...f, taskType: 'classification' }))}
                  style={{
                    padding: '0.85rem 1rem',
                    borderRadius: '10px',
                    border: form.taskType === 'classification'
                      ? '2px solid var(--accent)'
                      : '1.5px solid var(--border)',
                    background: form.taskType === 'classification'
                      ? 'rgba(67, 56, 202, 0.08)'
                      : 'var(--bg-card)',
                    boxShadow: form.taskType === 'classification'
                      ? '0 0 0 1px var(--accent), 0 2px 8px rgba(67, 56, 202, 0.15)'
                      : 'var(--neu-raised-sm)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Target size={16} color={form.taskType === 'classification' ? 'var(--accent)' : 'var(--text-muted)'} />
                      <span style={{
                        fontWeight: 700,
                        fontSize: '0.9rem',
                        color: form.taskType === 'classification' ? 'var(--accent)' : 'var(--text-primary)',
                      }}>
                        Classification
                      </span>
                    </div>
                    {form.taskType === 'classification' ? (
                      <CheckCircle2 size={16} color="var(--accent)" />
                    ) : (
                      <div style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px solid #94a3b8' }} />
                    )}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    Prediksi kategori / label diskrit (misal: Survived/Dead, Churn/Stay)
                  </div>
                </div>

                {/* Regression Radio Card */}
                <div
                  onClick={() => setForm(f => ({ ...f, taskType: 'regression' }))}
                  style={{
                    padding: '0.85rem 1rem',
                    borderRadius: '10px',
                    border: form.taskType === 'regression'
                      ? '2px solid var(--accent)'
                      : '1.5px solid var(--border)',
                    background: form.taskType === 'regression'
                      ? 'rgba(67, 56, 202, 0.08)'
                      : 'var(--bg-card)',
                    boxShadow: form.taskType === 'regression'
                      ? '0 0 0 1px var(--accent), 0 2px 8px rgba(67, 56, 202, 0.15)'
                      : 'var(--neu-raised-sm)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <TrendingUp size={16} color={form.taskType === 'regression' ? 'var(--accent)' : 'var(--text-muted)'} />
                      <span style={{
                        fontWeight: 700,
                        fontSize: '0.9rem',
                        color: form.taskType === 'regression' ? 'var(--accent)' : 'var(--text-primary)',
                      }}>
                        Regression
                      </span>
                    </div>
                    {form.taskType === 'regression' ? (
                      <CheckCircle2 size={16} color="var(--accent)" />
                    ) : (
                      <div style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px solid #94a3b8' }} />
                    )}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    Prediksi angka kontinu (misal: Harga Rumah, Usia, Gaji)
                  </div>
                </div>
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label className="label">Description (optional)</label>
              <textarea
                className="input"
                placeholder="What is this project about?"
                value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                rows={2}
              />
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              <button className="btn btn-secondary" onClick={() => setShowModal(false)}>
                Cancel
              </button>
              <button
                id="create-project-btn"
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={creating || !form.name.trim()}
              >
                {creating ? <span className="spinner" /> : <Plus size={16} />}
                Create Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
