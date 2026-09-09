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
            boxShadow: '0 8px 24px rgba(99, 102, 241, 0.4)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            border: 'none',
          }}
        >
          <Sparkles size={16} />
          <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>AI Assistant</span>
          <ChevronUp size={14} />
        </button>
      ) : (
        <div
          className="animate-slide-up"
          style={{
            width: '340px',
            background: '#181825',
            border: '1px solid #313244',
            borderRadius: '16px',
            boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
            padding: '1.25rem',
            color: '#cdd6f4',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{
                width: 28, height: 28, borderRadius: 8,
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Bot size={16} color="white" />
              </div>
              <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>AutoML AI Guide</span>
            </div>
            <button
              onClick={() => setOpen(false)}
              style={{ background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          {guidance ? (
            <div>
              <div style={{
                background: '#1e1e2e',
                borderRadius: '10px',
                padding: '0.85rem',
                border: '1px solid #313244',
                marginBottom: '0.85rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#cba6f7', fontSize: '0.78rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                  <CheckCircle size={12} />
                  {guidance.step_title}
                </div>
                <p style={{ fontSize: '0.8rem', color: '#a6adc8', lineHeight: 1.4, margin: 0 }}>
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
                  padding: '0.6rem',
                  fontSize: '0.82rem',
                }}
              >
                {guidance.action_label} <ArrowRight size={14} />
              </button>
            </div>
          ) : (
            <p style={{ fontSize: '0.8rem', color: '#a6adc8' }}>Memuat panduan AI...</p>
          )}
        </div>
      )}
    </div>
  );
}
