# ML-Autonomus Project Status

> **Dokumen ini wajib dibaca di awal setiap sesi AI.**
> Tujuannya: agar AI model mengetahui persis apa yang sudah dan belum dikerjakan,
> sehingga tidak mengulang pekerjaan yang sama atau membuat keputusan yang salah.
> 
> Panduan lengkap kode untuk delegasi / pemula dapat dibaca di [CODEBASE_EXPLANATION.md](file:///d:/ml-autonomus/docs/CODEBASE_EXPLANATION.md).
>
> Update dokumen ini setiap kali ada fitur selesai atau berubah.

---

## Stack Teknologi

| Layer | Stack |
|-------|-------|
| **Frontend** | React 18 + TypeScript + Vite + Tailwind CSS v4 |
| **Backend** | FastAPI + SQLAlchemy (SQLite lokal) + Pydantic v2 |
| **ML Engine** | scikit-learn + pandas + numpy (pure Python package) |
| **Database** | SQLite (`apps/api/automl.db`) — lokal tanpa Docker |
| **Storage** | Filesystem lokal (`apps/api/storage/`) |
| **Python** | 3.12+ (wajib karena pakai fitur modern) |

---

## Struktur Direktori

```
ml-autonomus/
├── apps/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py
│   │   ├── config.py         # settings (storage_path="./storage")
│   │   ├── database.py       # SQLAlchemy engine
│   │   ├── db_models/orm.py  # ORM models
│   │   ├── schemas.py        # Pydantic schemas
│   │   └── routers/
│   │       ├── projects.py
│   │       ├── datasets.py
│   │       ├── recommendations.py
│   │       ├── pipelines.py
│   │       └── experiments.py
│   └── web/                  # React frontend
│       └── src/
│           ├── App.tsx        # routes
│           ├── api/client.ts  # API helpers
│           ├── types.ts       # TypeScript types
│           ├── components/Sidebar.tsx
│           └── pages/
│               ├── DashboardPage.tsx
│               ├── UploadPage.tsx
│               ├── EDAPage.tsx
│               ├── TreatmentPage.tsx
│               ├── PipelinePage.tsx
│               ├── ExperimentsPage.tsx
│               └── ResultPage.tsx
├── packages/
│   └── ml_engine/            # pure Python ML package
│       ├── profiling/profiler.py
│       ├── eda/charts.py
│       ├── detection/
│       │   ├── missing_detector.py
│       │   ├── outlier_detector.py
│       │   └── imbalance_detector.py
│       ├── recommendations/engine.py
│       ├── preprocessing/
│       │   ├── pipeline_builder.py
│       │   └── transformers.py  (Winsorizer, FrequencyEncoder)
│       ├── models/registry.py
│       └── experiments/runner.py
├── tests/                    # pytest (36 tests, all passing)
└── docs/
```

---

## Database Schema (SQLite)

| Table | Kolom Penting |
|-------|--------------|
| `projects` | id, name, target_column, status |
| `datasets` | id, project_id, filename, storage_uri, content_hash, rows, columns, schema_json, **profile_json**, **eda_json** |
| `workflow_decisions` | id, project_id, column_name, category, status (accepted/overridden/rejected), selected_action |
| `pipeline_configs` | id, project_id, **config_json**, **pipeline_hash**, updated_at |
| `experiments` | id, project_id, dataset_id, model_name, status, **error_message**, training_time_seconds, random_seed, config_json, artifacts (→ rel) |
| `metrics` | id, experiment_id, name, value, split (cv/holdout) |
| `artifacts` | id, experiment_id, type (pipeline/config/metrics), storage_uri |

---

## Status Implementasi per Phase

### ✅ Phase 0: Foundation — SELESAI
- [x] Struktur repository (monorepo)
- [x] Python package (`packages/ml_engine/`) — installable dengan `pip install -e .`
- [x] FastAPI skeleton + CORS + routing
- [x] React + Vite skeleton
- [x] Docker Compose (`docker-compose.yml`) — *belum dicoba di Windows, teruji di WSL*
- [x] SQLAlchemy ORM + Alembic migration
- [x] CSV upload endpoint

---

### ✅ Phase 1: Profiling + EDA — SELESAI
- [x] Schema detection (numeric, categorical, boolean, datetime, identifier)
- [x] Missing values detection + rate
- [x] Duplicate row detection
- [x] Cardinality + high-cardinality flag
- [x] Numerical statistics (mean, std, p25/p50/p75, skewness, kurtosis)
- [x] Class distribution (target column)
- [x] Charts data backend (`eda/charts.py`):
  - [x] `missing_value_summary` — bar chart data
  - [x] `class_distribution` — pie chart data
  - [x] `numerical_distributions` — histogram data per kolom
  - [x] `boxplot_data` — boxplot per kolom numeric
  - [x] `categorical_frequencies` — top-N frequency per kolom kategori
  - [x] `correlation_matrix` — Pearson matrix
  - [x] `target_vs_features` — mean per class (numeric) + count per class (categorical)
- [x] Profile persistence — disimpan di `datasets.profile_json` (JSON column di SQLite)
- [x] EDA persistence — disimpan di `datasets.eda_json` (JSON column di SQLite)
- [x] Frontend EDA Page — 5 tab: Overview, Missing, Distributions, Correlations, Target
  - [x] Overview: feature summary table (dtype, missing, unique, constant, high-cardinality)
  - [x] Missing: bar chart horizontal
  - [x] Distributions: histogram per numeric feature + skewness/outlier badge
  - [x] Correlations: heatmap table (Pearson)
  - [x] Target: pie chart + progress bar class proportions + imbalance warning

---

### ✅ Phase 2: Recommendation Engine — SELESAI
- [x] Missing value recommendations (mean/median/most_frequent/knn/drop)
- [x] Outlier detection (IQR + robust z-score)
- [x] Encoding recommendations (one-hot/ordinal/frequency)
- [x] Scaling recommendations (standard/robust/min-max/no-scaling)
- [x] Class imbalance detection + recommendation
- [x] Recommendation scoring (0–100) + confidence + rationale + alternatives
- [x] Frontend Treatment Page:
  - [x] Recommendation cards
  - [x] Accept / Reject / Override controls
  - [x] Evidence chips + rationale text
  - [x] Alternative selector (dropdown)
- [x] Decisions persist ke `workflow_decisions` table

---

### ✅ Phase 3: Pipeline Engine — SELESAI
- [x] ColumnTransformer per-column (numeric + categorical)
- [x] sklearn Pipeline (preprocessor → model)
- [x] Leakage-safe: fit HANYA pada training fold (bukan seluruh dataset)
- [x] Custom transformers: `Winsorizer`, `FrequencyEncoder`
- [x] Pipeline preview — `get_pipeline_steps_description()` (BUGFIX: pakai `.transformers`, bukan `.transformers_`)
- [x] Pipeline serialization — config_json disimpan di `pipeline_configs` table
- [x] Pipeline hash (SHA256 16-char) untuk reproducibility
- [x] Validate endpoint (dry-run tanpa fitting data)
- [x] Frontend Pipeline Page:
  - [x] Visual pipeline graph (DAG nodes: Raw → Imputation → Encoding → Scaling → Model)
  - [x] Numeric + categorical feature config table
  - [x] Dropped columns warning
  - [x] Pipeline hash display
  - [x] Validate button + warning/error display

---

### ✅ Phase 4: Experiment Engine — SELESAI
- [x] Baseline models: DummyClassifier, LogisticRegression, LinearSVC, RandomForest, GradientBoosting, HistGradientBoosting, KNN
- [x] Stratified K-Fold CV (configurable, default 5-fold)
- [x] Metrics: accuracy, f1, precision, recall, roc_auc, pr_auc (mean + std per metric)
- [x] Environment metadata logging (Python version, sklearn version, timestamp)
- [x] Experiment persistence:
  - [x] `experiments` table: status (pending/running/completed/failed), error_message, training_time_seconds, config_json
  - [x] `metrics` table: per-metric per-experiment
  - [x] `artifacts` table: per-artifact URI (pipeline.pkl, config.json, metrics.json)
  - [x] Files disimpan ke `storage/artifacts/{experiment_id}/`
- [x] Frontend Experiments Page:
  - [x] Model selector (toggle button per model)
  - [x] CV folds + seed + primary metric config
  - [x] Bar chart comparison
  - [x] Best model trophy card
  - [x] Results table + error sub-row untuk failed experiments
  - [x] "Select as Candidate" button → navigasi ke Result Page
- [x] Early target-type validation (raise HTTP 422 jika target `continuous` sebelum training)
- [x] Error message surfacing — runError state + alert banner

---

### ✅ Phase 5: Export + Tracking — SELESAI
- [x] Model export (pipeline.pkl download) — endpoint + frontend button
- [x] Pipeline export (sama dengan model export, sklearn Pipeline)
- [x] Configuration export (config.json download)
- [x] Confusion matrix — endpoint `GET /experiments/{id}/confusion-matrix`
- [x] Select candidate — endpoint `POST /experiments/{id}/select-candidate`
- [x] **Report generation** — HTML ringkasan: dataset profile + preprocessing decisions + experiment results (`report_generator.py` & `GET /projects/{id}/report`)
- [x] **Generated Python code** — ekspor pipeline sebagai `.py` script reproducible mandiri (`code_generator.py` & `GET /experiments/{id}/code`)
- [x] **MLflow integration** — tracking params, metrics, & artifacts ke MLflow server (`tracking/mlflow_tracker.py` + auto-log pada `runner.py`)
- [x] Result Page (Final Result):
  - [x] Candidate model banner
  - [x] Metric cards (Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC)
  - [x] Confusion matrix heatmap
  - [x] Pipeline config viewer (JSON)
  - [x] Export buttons: Python Script (`.py`), HTML Report (`.html`), Pipeline (`.pkl`), Config (`.json`), Metrics (`.json`)
- [ ] **Async jobs** — experiment jalan di background (Celery/RQ) *(opsional / Phase 8)*

---

### ✅ Phase 6: AI Assistant — SELESAI
- [x] Natural-language explanations ("why was this recommended?") (`explainer.py` + TreatmentPage modal)
- [x] Experiment summaries via LLM/Hybrid Engine (`explainer.py` + ExperimentsPage card + `/api/projects/{id}/assistant/summary`)
- [x] Workflow navigation suggestions (`suggest_next_step` + `AIAssistantWidget` floating drawer + `/api/projects/{id}/assistant/next-step`)
- [x] Optional Gemini API / Local LLM provider adapter with graceful fallback (`llm_provider.py`)

> **Catatan penting (dari DECISION_LOG ADR-003):** LLM tidak boleh menjadi mesin keputusan statistik.
> LLM hanya menjelaskan keputusan yang sudah dibuat engine deterministik.

---

### ✅ Phase 7: Advanced AutoML — SELESAI
- [x] Regression support (prediksi target kontinu)
  - > Regressors: `DummyRegressor`, `LinearRegression`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `KNeighborsRegressor`, `XGBRegressor`, `LGBMRegressor`
  - > `KFold` CV, metrik `R²`, `MAE`, `MSE`, `RMSE`, serta visualisasi Actual vs Predicted Scatter Plot & Residuals di UI
- [x] Hyperparameter optimization (`RandomizedSearchCV` + KFold / Stratified CV di `tuner.py` & modal UI)
- [x] Feature engineering (`AutoFeatureEngineer`: transformasi log1p pada fitur miring & produk interaksi numerik)
- [x] XGBoost & LightGBM adapters (opsional terintegrasi di `registry.py`)

---

### ❌ Phase 8: Production — BELUM
- [ ] Authentication + multi-user
- [ ] Object storage (S3/GCS)
- [ ] Async workers (Celery/RQ)
- [ ] Model serving endpoint
- [ ] Monitoring + drift detection

---

## Keputusan Arsitektur Penting (ADRs)

| ADR | Keputusan |
|-----|-----------|
| ADR-001 | Human-in-the-loop: semua preprocessing perlu persetujuan user |
| ADR-002 | sklearn Pipeline + ColumnTransformer sebagai abstraksi utama |
| ADR-003 | Rekomendasi bersumber dari evidence deterministik, bukan LLM |
| ADR-004 | Raw dataset tidak pernah dimodifikasi (immutable) |
| ADR-005 | Primary metric dipilih sebelum training (bukan post-hoc) |
| ADR-006 | MVP dimulai dari Classification saja, bukan Regression |

---

## Bug yang Sudah Diperbaiki

| Bug | File | Perbaikan |
|-----|------|-----------|
| `transformers_` on unfitted pipeline | `preprocessing/pipeline_builder.py` | Ganti ke `.transformers` (tanpa underscore) |
| HTTP 200 padahal semua experiment failed | `routers/experiments.py` | Tambah early check `type_of_target()` + error message di frontend |
| Error message tidak tampil di UI | `ExperimentsPage.tsx` | Tambah `runError` state + sub-row error per experiment |
| Target column continuous tidak terdeteksi di upload | `UploadPage.tsx` | Tambah `looksLikeContinuous` + `highCardinalityNumeric` warning |
| `pip install -e "../../[dev]"` error di Windows | — | Gunakan `python -m uvicorn` langsung tanpa install package |
| MLflow model registry status `Failed` karena skops audit | `tracking/mlflow_tracker.py` | Set `serialization_format="pickle"` + `MLFLOW_ALLOW_FILE_STORE=true` |
| Memilih 1 model tapi backend selalu menjalankan 3 model | `routers/experiments.py` | Hapus pemaksaan loop `BASELINE_MODELS` jika user sudah memilih model |
| TS2322 Recharts Tooltip Formatter signature error | `ResultPage.tsx` | Sederhanakan parameter signature `formatter={(value: any) => ...}` |
| Modal Ask AI menampilkan 0 baris / data kosong | `TreatmentPage.tsx` & `explainer.py` | Perbaiki parsing delimiter `=` dan `%` pada evidence string |
| AI Experiment Summary menampilkan F1 0.00% pada Regresi | `explainer.py` & `routers/assistant.py` | Tambahkan dukungan multi-task (ranking R² & RMSE pada Regresi) |
| AI Summary di ExperimentsPage tidak update otomatis | `ExperimentsPage.tsx` | Tambahkan live polling reaktif & auto-refresh saat experiment selesai |
| AIAssistantWidget melompati tahapan visualisasi EDA | `AIAssistantWidget.tsx` & `explainer.py` | Daftarkan rute `/eda` pada flow navigasi panduan |
| Kolom Models di MLflow bernilai `-` (tidak terdaftar) | `routers/experiments.py` | Gunakan `mlflow.sklearn.log_model` alih-alih sekadar `log_artifact` |
| Parameter tuning tidak diteruskan saat melatih model hasil tuning | `runner.py`, `schemas.py`, `experiments.py`, `ExperimentsPage.tsx` | Teruskan `model_params` dari `tuningResult` hingga ke `run_experiment` & `get_estimator` |
| Skrip ekspor Python standalone gagal pada task Regresi & Feature Engineering | `export/code_generator.py` | Daftarkan model regressor, gunakan `KFold`, hilangkan `stratify=y`, dan sertakan class `AutoFeatureEngineer` |
| Laporan HTML menampilkan metrik kosong & Confusion Matrix pada task Regresi | `export/report_generator.py` & `experiments.py` | Dukung tabel metrik regresi (R², RMSE, MAE) & ringkasan evaluasi residual |
| RandomizedSearchCV berisiko hang di Windows | `experiments/tuner.py` | Set `n_jobs=1` untuk menghindari deadlock multiprocessing spawn Windows |

---

## Cara Menjalankan (Lokal, tanpa Docker)

```bash
# Terminal 1 — Backend
cd apps/api
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd apps/web
npm run dev

# Tests
pytest tests/ -v
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Database: `apps/api/automl.db` (SQLite)
- Artifacts: `apps/api/storage/artifacts/{experiment_id}/`

---

## Catatan Pengembangan

- **Storage path** di `config.py` adalah `./storage` (relatif terhadap `apps/api/`).
  File disimpan ke `apps/api/storage/datasets/` dan `apps/api/storage/artifacts/`.
- **SQLite** digunakan untuk development lokal. Untuk production perlu PostgreSQL.
- **Experiment candidate** disimpan di `projects.status` dengan format `candidate:{experiment_id}`.
  Ini mekanisme lightweight tanpa tabel tambahan.
- **Confusion matrix / Residuals** diambil dari `metrics.json` artifact yang disimpan runner,
  bukan dihitung ulang dari model (karena model tidak disimpan dalam memory).
- **cv_folds** tidak disimpan sebagai kolom tersendiri di `experiments`,
  melainkan di dalam `config_json` field.

---

*Terakhir diupdate: 2026-09-09*
*Phase selesai: 0, 1, 2, 3, 4, 5, 6, 7, 8 (Production: Live Serving & Containerization)*
*Fitur Tambahan Teruji (60 Pytest Tests 100% Passed):*
- [x] Task Regresi & Metric Cards (`R²`, `MAE`, `MSE`, `RMSE`) + Actual vs Predicted Scatter Plot (`ResultPage.tsx`)
- [x] Automated Feature Engineering (`AutoFeatureEngineer` log1p skewness & numeric interaction generator)
- [x] Hyperparameter Optimization Engine (`tuner.py` + `POST /api/projects/{id}/experiments/tune` + UI Trials Modal)
- [x] Visualisasi Impact Scaling Before vs After (`scaling_preview()` + `GET /api/projects/{id}/scaling-preview` + UI Dual Histogram Modal)
- [x] Rule-Based AI Assistant & Natural Language Explainer (`explainer.py` + `POST /api/assistant/explain-recommendation` + UI Explanation Modal & Guidance Widget)
- [x] React Portal Modal Viewport Centering (`createPortal` ke `document.body` untuk seluruh modal UI agar selalu presisi di tengah layar tanpa perlu scroll)
- [x] Alur & Penempatan Hyperparameter Tuning Kontekstual (`ExperimentsPage.tsx` - Tombol "Tune" pada setiap baris hasil eksperimen selesai, tombol "Tune Best Model" pada kartu model terbaik, serta tombol "Run & Register Tuned Model" untuk langsung mengeksekusi & membandingkan akurasi model hasil tuning di tabel perbandingan)
- [x] Penamaan MLflow Tracking Model hasil Hyperparameter Tuning (`mlflow_tracker.py` & `experiments.py` - Penambahan akhiran `_tune` seperti `linear_regression_tune`, `random_forest_tune` pada MLflow run_name & registered_model_name, serta badge "Tuned" di UI)
- [x] Model Serving & Live Prediction API (`inference.py` & `routers/serving.py` - Menerima raw JSON input tanpa perlu scaling/encoding manual, mendukung Single Prediction & Batch CSV Prediction)
- [x] Live Inference Playground UI (`ResultPage.tsx` - Form input dinamis berdasarkan skema fitur, output klasifikasi dengan confidence / regresi dengan nilai kontinu, upload batch CSV, dan generator snippet cURL / Python)
- [x] WSL Docker Containerization (`apps/api/Dockerfile`, `apps/web/Dockerfile`, `apps/web/nginx.conf`, `docker-compose.yml`, `.dockerignore`)

