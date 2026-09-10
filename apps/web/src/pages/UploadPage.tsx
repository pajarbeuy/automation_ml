import { useState, useCallback, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Upload, FileText, ChevronRight, AlertCircle, AlertTriangle, TrendingUp, Target } from 'lucide-react';
import { uploadDataset, listDatasets, getProject } from '../api/client';
import type { Dataset, Project } from '../types';

export function UploadPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string[][]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [existing, setExisting] = useState<Dataset[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  // Check if preview values for the selected target look like continuous numbers
  const targetColIdx = headers.indexOf(targetColumn);
  const targetPreviewValues = preview.map(row => row[targetColIdx]).filter(Boolean);
  const looksLikeContinuous = targetPreviewValues.length > 0 && targetPreviewValues.every(v => {
    const n = Number(v);
    return !isNaN(n);
  });
  const uniqueTargetValues = new Set(targetPreviewValues);
  const highCardinalityNumeric = targetPreviewValues.length > 0
    && uniqueTargetValues.size > 10
    && [...uniqueTargetValues].every(v => !isNaN(Number(v)));

  useEffect(() => {
    if (!projectId) return;
    Promise.all([
      getProject(projectId).catch(() => null),
      listDatasets(projectId).catch(() => []),
    ]).then(([proj, datasets]) => {
      if (proj) setProject(proj);
      if (datasets) setExisting(datasets);
    });
  }, [projectId]);

  const parseCSVPreview = (text: string) => {
    const lines = text.split('\n').slice(0, 6).filter(Boolean);
    const rows = lines.map(l => l.split(',').map(v => v.trim().replace(/^"|"$/g, '')));
    if (rows.length > 0) {
      setHeaders(rows[0]);
      setPreview(rows.slice(1));
      setTargetColumn(rows[0][rows[0].length - 1]);
    }
  };

  const handleFile = (f: File) => {
    if (!f.name.endsWith('.csv')) {
      setError('Only CSV files are supported.');
      return;
    }
    setFile(f);
    setError(null);
    const reader = new FileReader();
    reader.onload = e => parseCSVPreview(e.target?.result as string);
    reader.readAsText(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleUpload = async () => {
    if (!file || !targetColumn || !projectId) return;
    setUploading(true);
    setError(null);
    try {
      await uploadDataset(projectId, file, targetColumn);
      navigate(`/projects/${projectId}/eda`);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const taskType = project?.task_type || 'classification';
  const isRegression = taskType === 'regression';

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">
            Dataset Upload ({isRegression ? 'Regression Project' : 'Classification Project'})
          </h1>
          <p className="page-subtitle">Upload a CSV file and select the target column to predict</p>
        </div>
      </div>

      {/* Existing datasets */}
      {existing.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <h3 style={{ margin: 0 }}>Existing Datasets</h3>
          </div>
          {existing.map(d => (
            <div key={d.id} style={{
              display: 'flex', alignItems: 'center', gap: '0.75rem',
              padding: '0.85rem 1rem', background: 'var(--bg-inset)', borderRadius: '8px',
              border: '1px solid var(--border-inset)', boxShadow: 'var(--neu-inset-sm)', marginBottom: '0.5rem',
            }}>
              <FileText size={18} color="var(--accent)" />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)' }}>{d.filename}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  {d.rows?.toLocaleString()} rows · {d.columns} cols
                </div>
              </div>
              <button
                className="btn btn-primary btn-sm"
                onClick={() => navigate(`/projects/${projectId}/eda`)}
              >
                Continue <ChevronRight size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="grid-2" style={{ gap: '1.5rem', alignItems: 'stretch' }}>
        {/* Drop zone */}
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div
            id="upload-dropzone"
            className={`upload-zone ${dragging ? 'dragging' : ''}`}
            style={{ flex: 1, minHeight: '260px' }}
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onClick={() => fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <div style={{ marginBottom: '1rem' }}>
              <Upload size={40} color="var(--accent)" strokeWidth={1.5} />
            </div>
            {file ? (
              <>
                <div style={{ fontWeight: 600, color: 'var(--accent-hover)', marginBottom: '0.25rem' }}>
                  {file.name}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  {(file.size / 1024).toFixed(1)} KB — click to replace
                </div>
              </>
            ) : (
              <>
                <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                  Drop your CSV file here
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  or click to browse
                </div>
              </>
            )}
          </div>

          {error && (
            <div className="alert alert-danger" style={{ marginTop: '1rem' }}>
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>

        {/* Config panel */}
        <div className="card" style={{ height: '100%', minHeight: '260px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <h3 style={{ margin: 0 }}>Configuration</h3>
              <span className={`badge ${isRegression ? 'badge-success' : 'badge-primary'}`} style={{ textTransform: 'capitalize' }}>
                {isRegression ? <TrendingUp size={12} /> : <Target size={12} />} {taskType}
              </span>
            </div>

            {headers.length > 0 ? (
              <>
                <div className="form-group" style={{ marginBottom: '1.25rem' }}>
                  <label className="label" htmlFor="target-select">Target Column *</label>
                  <select
                    id="target-select"
                    className="input select"
                    value={targetColumn}
                    onChange={e => setTargetColumn(e.target.value)}
                  >
                    {headers.map(h => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                    The column the model will predict
                  </div>

                  {/* Target Column Info/Warning based on task_type */}
                  {isRegression && (looksLikeContinuous || highCardinalityNumeric) && (
                    <div className="alert alert-info" style={{ marginTop: '0.75rem', fontSize: '0.78rem' }}>
                      <TrendingUp size={14} />
                      <div>
                        <strong style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}><Target size={14} /> Continuous numeric target detected for Regression Project.</strong><br />
                        System is ready to train regression models (Linear Regression, Ridge, Random Forest, XGBoost, etc.) to predict <code>{targetColumn}</code>.
                      </div>
                    </div>
                  )}

                  {!isRegression && (looksLikeContinuous || highCardinalityNumeric) && (
                    <div className="alert alert-warning" style={{ marginTop: '0.75rem', fontSize: '0.78rem' }}>
                      <AlertTriangle size={14} />
                      <div>
                        <strong style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}><AlertTriangle size={14} /> Continuous numeric target detected.</strong><br />
                        This project is set to <strong>Classification</strong>. If you want to predict continuous values like prices or ages, please create a new project with Task Type set to <strong>Regression</strong> in the Dashboard.
                      </div>
                    </div>
                  )}
                </div>

                <div style={{ marginBottom: '1.25rem' }}>
                  <div className="label" style={{ marginBottom: '0.5rem' }}>Detected Columns</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', maxHeight: '120px', overflowY: 'auto' }}>
                    {headers.map(h => (
                      <span
                        key={h}
                        className={`badge ${h === targetColumn ? 'badge-primary' : 'badge-muted'}`}
                      >
                        {h === targetColumn && <Target size={11} style={{ marginRight: 2 }} />}
                        {h}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
                <FileText size={38} strokeWidth={1.5} style={{ marginBottom: '0.75rem', opacity: 0.5, color: 'var(--accent)' }} />
                <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.25rem', fontSize: '0.95rem' }}>No Dataset Uploaded</div>
                <div style={{ fontSize: '0.82rem', maxWidth: '280px' }}>Drop a CSV file on the left to configure target column and start automated profiling.</div>
              </div>
            )}
          </div>

          {headers.length > 0 && (
            <button
              id="upload-btn"
              className="btn btn-primary"
              style={{ width: '100%', marginTop: '1rem' }}
              onClick={handleUpload}
              disabled={uploading || !targetColumn}
            >
              {uploading ? (
                <>
                  <span className="spinner" />
                  Profiling dataset…
                </>
              ) : (
                <>
                  <Upload size={16} />
                  Upload & Profile Dataset
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* CSV Preview */}
      {preview.length > 0 && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Preview (first 5 rows)</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {headers.map(h => (
                    <th key={h} style={{ color: h === targetColumn ? 'var(--accent-hover)' : undefined }}>
                      {h === targetColumn && <Target size={13} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle', color: 'var(--accent-hover)' }} />}{h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                        {cell || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
