import { useNavigate, useParams, useLocation } from 'react-router-dom';
import {
  Upload, BarChart2, Sliders,
  GitBranch, FlaskConical, ChevronLeft, Brain, Trophy,
} from 'lucide-react';

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

export function Sidebar() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  const navItems: NavItem[] = [
    { label: 'Dataset Upload', path: 'upload', icon: <Upload size={16} /> },
    { label: 'EDA Workspace', path: 'eda', icon: <BarChart2 size={16} /> },
    { label: 'Data Treatment', path: 'treatment', icon: <Sliders size={16} /> },
    { label: 'Pipeline', path: 'pipeline', icon: <GitBranch size={16} /> },
    { label: 'Experiments', path: 'experiments', icon: <FlaskConical size={16} /> },
    { label: 'Final Result', path: 'result', icon: <Trophy size={16} /> },
  ];

  const isActive = (path: string) =>
    location.pathname.includes(`/${path}`);

  return (
    <div className="sidebar animate-fade-in">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Brain size={20} color="white" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>AutoML HIL</div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Human-in-the-Loop</div>
        </div>
      </div>

      {/* Back to Dashboard */}
      <div style={{ padding: '0.75rem' }}>
        <button
          className="btn btn-ghost"
          style={{ width: '100%', justifyContent: 'flex-start', gap: '0.5rem' }}
          onClick={() => navigate('/')}
        >
          <ChevronLeft size={14} />
          All Projects
        </button>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-section-title">Workflow</div>
        {navItems.map((item) => (
          <button
            key={item.path}
            className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
            onClick={() => navigate(`/projects/${projectId}/${item.path}`)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {/* Phase indicator */}
      <div style={{
        padding: '1rem',
        borderTop: '1px solid var(--border)',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
      }}>
        <div style={{ marginBottom: '0.5rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
          MVP v0.1.0 — Classification
        </div>
        <div>Follow the workflow steps in order for best results.</div>
      </div>
    </div>
  );
}
