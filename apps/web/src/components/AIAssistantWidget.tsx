import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Bot, Sparkles, ArrowRight, X, ChevronUp, CheckCircle } from 'lucide-react';
import { getNextStepGuidance } from '../api/client';

interface NextStepInfo {
  step_title: string;
  message: string;
  next_url: string;
  action_label: string;
}

export function AIAssistantWidget({ projectId }: { projectId?: string }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [guidance, setGuidance] = useState<NextStepInfo | null>(null);

  const getCurrentPageKey = useCallback(() => {
    const path = location.pathname;
    if (path.includes('/eda')) return 'eda';
    if (path.includes('/treatment')) return 'treatment';
    if (path.includes('/pipeline')) return 'pipeline';
    if (path.includes('/experiments')) return 'experiments';
    if (path.includes('/result')) return 'result';
    return 'upload';
  }, [location.pathname]);

  useEffect(() => {
    if (!projectId) return;
    const pageKey = getCurrentPageKey();
    getNextStepGuidance(projectId, pageKey)
      .then(res => setGuidance(res as NextStepInfo))
      .catch(() => setGuidance(null));
  }, [projectId, getCurrentPageKey]);

  if (!projectId) return null;

  return (
    <div style={{ position: 'fixed', bottom: '1.5rem', right: '1.5rem', zIndex: 99 }}>
      {!open ? (
        <button
          onClick={() => setOpen(true)}
          className="btn btn-primary"
          style={{
            borderRadius: '24px',
            padding: '0.65rem 1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Sparkles size={16} />
          <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>AI Assistant</span>
          <ChevronUp size={14} />
        </button>
      ) : (
        <div
          className="animate-slide-up card"
          style={{
            width: '340px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-highlight)',
            borderRadius: '16px',
            boxShadow: 'var(--neu-raised-hover)',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{
                width: 30, height: 30, borderRadius: 8,
                background: 'var(--bg-inset)',
                boxShadow: 'var(--neu-inset-sm)',
                border: '1px solid var(--border-inset)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--accent-hover)',
              }}>
                <Bot size={16} />
              </div>
              <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>AutoML AI Guide</span>
            </div>
            <button
              onClick={() => setOpen(false)}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          {guidance ? (
            <div>
              <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '10px',
                padding: '0.85rem',
                border: '1px solid var(--border-inset)',
                boxShadow: 'var(--neu-inset-sm)',
                marginBottom: '0.85rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-hover)', fontSize: '0.78rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                  <CheckCircle size={12} />
                  {guidance.step_title}
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                  {guidance.message}
                </p>
              </div>

              <button
                className="btn btn-primary"
                onClick={() => {
                  const targetPath = guidance.next_url.startsWith('/') ? guidance.next_url : `/${guidance.next_url}`;
                  navigate(`/projects/${projectId}${targetPath}`);
                  setOpen(false);
                }}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  padding: '0.65rem',
                  fontSize: '0.82rem',
                }}
              >
                {guidance.action_label} <ArrowRight size={14} />
              </button>
            </div>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Memuat panduan AI...</p>
          )}
        </div>
      )}
    </div>
  );
}
