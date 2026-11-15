import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Brain, Lock, Mail, User as UserIcon, Shield, ArrowRight, CheckCircle2, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { getErrorMessage } from '../api/client';

interface LocationState {
  from?: {
    pathname?: string;
  };
}

export const LoginPage: React.FC = () => {
  const { login, register, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'data_scientist' | 'viewer' | 'admin'>('data_scientist');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const locationState = location.state as LocationState | null;
  const redirectPath = locationState?.from?.pathname || '/';
  // If already authenticated, redirect
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate(redirectPath, { replace: true });
    }
  }, [isAuthenticated, navigate, redirectPath]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setSuccessMsg(null);
    setIsSubmitting(true);

    try {
      if (mode === 'login') {
        await login({ email, password });
        navigate(redirectPath, { replace: true });
      } else {
        await register({
          email,
          username: username.trim(),
          password,
          role,
        });
        setSuccessMsg('Pendaftaran berhasil! Silakan masuk menggunakan akun baru Anda.');
        setMode('login');
        setPassword('');
      }
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFillDemoAdmin = () => {
    setEmail('admin@automl.local');
    setPassword('admin123');
    setMode('login');
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center p-4 bg-[#e6ecf4]">
      <div className="w-full max-w-md card-elevated p-6 sm:p-8 rounded-2xl shadow-xl border border-[#d1d9e6]">
        {/* Brand Header */}
        <div className="flex flex-col items-center text-center mb-6">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#4338ca] text-white shadow-md mb-3">
            <Brain size={32} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-800">ML-Autonomus</h1>
          <p className="text-xs text-slate-500 mt-1">Autonomous & Human-in-the-Loop AutoML Platform</p>
        </div>

        {/* Mode Toggle Tabs */}
        <div className="flex rounded-xl bg-[#dde5ef] p-1 mb-6 border border-[#cbd5e1]">
          <button
            type="button"
            onClick={() => {
              setMode('login');
              setErrorMsg(null);
              setSuccessMsg(null);
            }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              mode === 'login'
                ? 'bg-white text-indigo-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Masuk (Login)
          </button>
          <button
            type="button"
            onClick={() => {
              setMode('register');
              setErrorMsg(null);
              setSuccessMsg(null);
            }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              mode === 'register'
                ? 'bg-white text-indigo-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Daftar Akun Baru
          </button>
        </div>

        {/* Alerts */}
        {errorMsg && (
          <div className="mb-4 flex items-start gap-2 rounded-xl bg-red-50 p-3 text-xs text-red-700 border border-red-200">
            <AlertCircle size={16} className="shrink-0 mt-0.5 text-red-500" />
            <p className="flex-1">{errorMsg}</p>
          </div>
        )}

        {successMsg && (
          <div className="mb-4 flex items-start gap-2 rounded-xl bg-emerald-50 p-3 text-xs text-emerald-800 border border-emerald-200">
            <CheckCircle2 size={16} className="shrink-0 mt-0.5 text-emerald-600" />
            <p className="flex-1">{successMsg}</p>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="form-group">
            <label className="text-xs font-semibold uppercase text-slate-600 flex items-center gap-1.5">
              <Mail size={13} className="text-indigo-600" /> Alamat Email
            </label>
            <input
              type="email"
              required
              placeholder="nama@perusahaan.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="input text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {mode === 'register' && (
            <>
              <div className="form-group">
                <label className="text-xs font-semibold uppercase text-slate-600 flex items-center gap-1.5">
                  <UserIcon size={13} className="text-indigo-600" /> Username
                </label>
                <input
                  type="text"
                  required
                  placeholder="username_anda"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  className="input text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="form-group">
                <label className="text-xs font-semibold uppercase text-slate-600 flex items-center gap-1.5">
                  <Shield size={13} className="text-indigo-600" /> Peran (Role)
                </label>
                <select
                  value={role}
                  onChange={e => setRole(e.target.value as 'data_scientist' | 'viewer' | 'admin')}
                  className="select text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 bg-[#dde5ef] focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="data_scientist">Data Scientist (Full Workspace Access)</option>
                  <option value="viewer">Viewer (Read-Only Access)</option>
                  <option value="admin">Administrator (Tenant & System Control)</option>
                </select>
              </div>
            </>
          )}

          <div className="form-group">
            <label className="text-xs font-semibold uppercase text-slate-600 flex items-center gap-1.5">
              <Lock size={13} className="text-indigo-600" /> Kata Sandi (Password)
            </label>
            <input
              type="password"
              required
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="input text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="btn btn-primary w-full py-2.5 mt-2 flex items-center justify-center gap-2 rounded-xl text-sm font-semibold transition-all shadow-md hover:shadow-lg disabled:opacity-50"
          >
            {isSubmitting ? (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            ) : (
              <>
                <span>{mode === 'login' ? 'Masuk ke Workspace' : 'Daftarkan Akun'}</span>
                <ArrowRight size={15} />
              </>
            )}
          </button>
        </form>

        {/* Quick Demo Hint */}
        {mode === 'login' && (
          <div className="mt-6 pt-4 border-t border-slate-300/60 text-center">
            <p className="text-xs text-slate-500 mb-2">Ingin mencoba langsung akun bawaan lokal?</p>
            <button
              type="button"
              onClick={handleFillDemoAdmin}
              className="text-xs font-medium text-indigo-700 hover:text-indigo-900 underline"
            >
              Gunakan Default Admin (admin@automl.local)
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
