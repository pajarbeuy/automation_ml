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
│       │   ├── imbalance_detector.py
│       │   └── leakage_detector.py  (NEW: target proxy + identifier detection)
│       ├── recommendations/
│       │   ├── engine.py
│       │   └── candidate_evaluator.py (Candidate Strategy Generation & Empirical CV Validation)
│       ├── preprocessing/
│       │   ├── pipeline_builder.py
│       │   └── transformers.py  (Winsorizer, FrequencyEncoder)
│       ├── models/registry.py
│       ├── experiments/runner.py
│       ├── serving/inference.py
│       └── monitoring/drift_detector.py
├── tests/                    # pytest (150 total tests, 100% passing)
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
- [x] Data leakage detection (Target proxy correlation $\ge 0.95$ & unique Identifier drop)
- [x] Recommendation scoring (0–100) + confidence + rationale + alternatives
- [x] **Candidate Treatment Generation** (`candidate_evaluator.py`):
  - Holistic candidate strategy generation: `recommended` (heuristics + leakage drop), `baseline` (simple raw impute + leakage drop), and `alternative` (robust/quantile scaling + alternative encodings).
- [x] **Empirical Recommendation Validation & Metric Lift Benchmark**:
  - Reference estimator fast 3-fold cross-validation (`LogisticRegression(max_iter=500)` / `Ridge()`).
  - Treatment lift calculation $\Delta = \text{Score}_{\text{rec}} - \text{Score}_{\text{base}}$ and validation status (`VALIDATED`, `NEUTRAL`, `SUBOPTIMAL`).
  - Strict leakage isolation: zero target leakage across folds.
- [x] **Backend API Endpoints**:
  - `GET /api/projects/{id}/treatment-candidates`
  - `POST /api/projects/{id}/treatment-candidates/evaluate`
  - `POST /api/projects/{id}/treatment-candidates/apply` (One-Click Batch Apply)
- [x] **Frontend Treatment Page (Macro & Micro Unification)**:
  - Macro Level: Strategy benchmark cards, empirical CV lift badge, reference model info, and 1-Click Batch Apply.
  - Micro Level: Granular per-column recommendation cards, evidence chips, and manual human-in-the-loop overrides.
  - State Sync & Integrity: Pemisahan tegas antara evaluasi benchmark (simulasi CV murni) dan *Apply Strategy* (penyimpanan atomik ke database). Tombol *Terapkan Strategi Ini* selalu interaktif di semua kandidat, serta konfirmasi otomatis saat *Build Pipeline* jika masih ada issue pending.
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
- [x] **Candidate Preprocessing Generation & Empirical Proxy Validation (Mode A)** (`candidate_evaluator.py`, `routers/recommendations.py`, `TreatmentPage.tsx`):
  - > 3 paket strategi pra-pemrosesan: `Recommended` (evidence-based), `Baseline` (minimal/standar), `Alternative` (robust/variasi)
  - > Perbandingan berpasangan sejati (True Paired CV) dengan pre-generated splits bersama (`KFold` / `StratifiedKFold`)
  - > Metodologi: Paired-fold CI screening berbasis selisih fold berpasangan ($d_i = \text{Rec}_i - \text{Base}_i$) dan interval kepercayaan dua sisi Student's t ($df = k - 1$)
  - > Wording jujur tanpa overclaim: Menghindari frasa "statistically proven" / "fully rigorous", merujuk secara ketat sebagai "paired-fold CI screening" dalam protokol proksi CV
  - > Status metodologis jujur: `SUPERIOR`, `INCONCLUSIVE`, `SUBOPTIMAL`, `INVALID_EVALUATION` (sign agreement sebagai bukti pendukung)
  - > Guard zero-baseline: `relative_difference` menghasilkan `None` (`rel: N/A`) saat baseline bernilai 0 untuk mencegah division by zero atau klaim lift semu
  - > Metadata eksplisit: `task_type`, `ci_method` (`paired_two_tailed_student_t`), `degrees_of_freedom`, `n_valid_folds`, `estimator`, `estimator_parameters`, `cv_strategy`, `n_splits`, `scoring_metric`, `confidence_level`
  - > Evaluasi auditable: raw fold scores (`baseline_fold_scores`, `recommended_fold_scores`, `fold_differences`)
  - > Penanganan edge cases eksplisit: $k < 2$, varians nol ($s_d < 1e-12$), NaN/Inf, kegagalan evaluasi
  - > Strict Option A: `class_weight=None` pada proxy estimator, tanpa heuristik ketimpangan kelas prematur
- [x] **Experiment Name & Pipeline Search Engine** (`ExperimentsPage.tsx`, `routers/experiments.py`, `orm.py`, `schemas.py`):
  - > Input `Experiment Name / Pipeline Tag (Opsional)` saat menjalankan eksperimen training maupun tuning
  - > Output representasi ganda: **Nama Algoritma + Badge Experiment Name** pada tabel dan chart
  - > Fitur Search bar real-time untuk memfilter daftar eksperimen berdasarkan algoritma, label eksperimen/pipeline, atau status run
  - > Backward-compatible persistence via `config_json["experiment_name"]` dan SQLAlchemy `@property` tanpa risiko migrasi SQLite database

---

### ✅ Phase 8: Production — SELESAI
- [x] Model serving endpoint (`serving/inference.py` + `routers/serving.py`)
- [x] Live Inference Playground UI (form dinamis, batch CSV, cURL/Python snippet)
- [x] Native Drift Monitoring (KS-Test, PSI, background logging)
- [x] Uncertainty Quantification regresi (R² confidence, ±MAE margin, prediction interval)
- [x] WSL Docker Containerization (Dockerfile backend + frontend + nginx + docker-compose)
- [x] Complete Regression Test Suite (157 total tests, 100% passing)
- [x] Authentication + multi-user & Multi-tenancy (JWT Bearer, API Keys, Tenant isolation, RBAC `admin`/`data_scientist`/`viewer`)
- [x] Cloud Object Storage (S3 / MinIO / Local adapter with AWS SigV4)
- [x] Distributed Asynchronous Job Workers (Task Queue, `JobWorker`, Background Thread/Process polling, job status & cancellation tracking)
- [x] Meta-Learning Recommendation Ranker (`MetaLearningRanker`, surrogate model, meta-feature extraction, cold-start blended scoring)

---

### ✅ Phase 9: Autonomous NLP & Sentiment Analysis — SELESAI
- [x] Text Profiler (lexical metrics, TTR, document length distribution, stopword ratio) (`profiling/text_profiler.py`)
- [x] Leakage-Safe Text Preprocessing (`TextCleanerTransformer`, `TfidfVectorizer` fit inside CV training fold only) (`preprocessing/text_transformers.py`)
- [x] NLP Model Registry (`MultinomialNB`, `ComplementNB`, `LogisticRegression`, `LinearSVC`, `SGDClassifier`) (`models/registry.py`)
- [x] Sentiment Word Attribution (Linear coefficients / log-probabilities top positive & negative highlights) (`assistant/text_explainer.py`)
- [x] Live Text Playground & Sentiment Serving API (`ResultPage.tsx` & `serving/inference.py`)
- [x] Text EDA Visualizer tab for lexical statistics, unigrams, and bigrams (`EDAPage.tsx`)
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
| ADR-007 | Candidate Treatment Evaluation berbasis Paired-Difference Student's t CI pada protokol proksi, bukan klaim keunggulan universal |

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
| `outlier_detector.py` menghitung ulang IQR yang sudah ada di profil | `detection/outlier_detector.py` | Gunakan `f.p25`, `f.p75`, `f.outlier_rate_iqr` dari profil; hanya robust z-score yang tetap dihitung dari DataFrame |
| `_recommend_scaling` tidak masuk branch StandardScaler saat skewness == 0.0 | `recommendations/engine.py` | Ubah `elif skewness and abs(...) < 0.5` → `elif skewness is not None and abs(...) < 0.5` |
| Threshold deteksi identifier string di profiler tidak konsisten dengan `leakage_detector.py` | `profiling/profiler.py` | Ubah kondisi dari `nunique() == len(dropna())` (100%) ke `unique_ratio >= 0.95` agar selaras dengan `IDENTIFIER_UNIQUE_RATIO_THRESHOLD` |
| `AutoFeatureEngineer` memilih pasangan interaksi dari korelasi tertinggi (memperparah multikolinearitas) | `preprocessing/feature_engineering.py` | Pilih pasangan berkorelasi rendah-sedang (0.1–0.7); urutkan mendekati 0.4 (paling complementary); fallback ke pasangan < 0.9 jika tidak ada kandidat |
| `interval_lower` menggunakan `max(0.0, pred - mae)` yang mengasumsikan target selalu non-negatif | `serving/inference.py` | Hapus `max(0.0, ...)` — interval bawah boleh negatif jika domain target memang mengizinkan |
| `target_vs_features` melempar KeyError jika `target_col` tidak ada atau None | `eda/charts.py` | Tambahkan guard `if not target_col or target_col not in df.columns: return {}` dan skip kolom target dari perbandingan fitur |
| `FrequencyEncoder` menganggap `np.nan` sebagai kategori valid | `preprocessing/transformers.py` & `export/code_generator.py` | Gunakan `pd.notna(col)` saat fit dan petakan nilai missing ke 0.0 saat transform |
| Keputusan `drop_column` dari kategori `data_leakage` terabaikan di pipeline builder | `preprocessing/pipeline_builder.py` | Perbaiki kondisi `any(action == "drop_column" for action in col_decisions.values())` sehingga kolom bocor dipastikan terhapus dari pipeline scikit-learn |
| AI Benchmark Summary membandingkan run model yang sama sebagai runner-up (selisih 0.0000) | `assistant/explainer.py` | Filter runner-up strictly dari arsitektur algoritma yang berbeda dan konversi nama teknis model menjadi Title Case profesional |
| Metrik $R^2$ pada kartu Best Performing Model bertolak belakang dengan tabel perbandingan | `ExperimentsPage.tsx` | Format metrik $R^2$ regresi konsisten sebagai angka desimal standar ML (`0.8273 ± 0.3092`) alih-alih persen ganda |
| Tanda asterisks markdown `**...**` muncul mentah di layar AI Benchmark Summary | `ExperimentsPage.tsx` | Tambahkan helper `renderMarkdownBold` untuk mengonversi sintaks bold menjadi elemen strong berbobot kontras tinggi |
| MLflow tracking timeout ketika `MLFLOW_TRACKING_URI` mengarah ke server offline | `tracking/mlflow_tracker.py` | Tambah `_is_server_reachable` socket check dengan timeout 0.5s sebelum request untuk mencegah test/runner hanging |

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

*Terakhir diupdate: 2026-11-11*
*Versi saat ini: **v1.2.0***
*Phase selesai: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (Autonomous NLP Sentiment Analysis, Distributed Async Jobs, Multi-Tenancy & Security, Cloud Storage, Meta-Learning)*
*Fitur Tambahan Teruji (**162 Fast Tests / 168 Total Tests 100% Passed**):*
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
- [x] Neumorphism (Soft UI) Design System & Lucide Icons Overhaul (`apps/web/src/index.css`, seluruh komponen & halaman - Menerapkan tema **Soft Light Clay Neumorphism** tanpa gradasi berlebih, shadow diperhalus tanpa kesan berlebihan/bengkak, kartu berukuran serasi & seimbang, logo otak solid purple `#7c3aed`, navigasi responsif tablet/mobile dengan hamburger drawer, serta eliminasi 100% Unicode emoji menjadi semantic SVG icons dari `lucide-react`)
- [x] End-to-End Error Handling Architecture (`ErrorBoundary.tsx` frontend error boundary, Axios response interceptor di `client.ts` untuk unwrap detail validasi Pydantic/FastAPI, dan FastAPI global exception handler di `main.py`)
- [x] Robustness & Edge-Case Hardening (`serving/inference.py`, `routers/serving.py`, `routers/datasets.py`, `test_robustness_serving.py`, `test_robustness_datasets.py`):
  - Validasi payload kosong (`{}` / `[]` / empty DataFrame) menghasilkan HTTP 422 deskriptif tanpa crash 500.
  - Skema inferensi mendeteksi kolom fitur yang hilang (*missing features*) dan melaporkan daftar kolom yang kurang secara spesifik.
  - Deteksi tipe data tidak valid (teks pada kolom numerik) menghasilkan pesan error yang ramah pengguna.
  - Toleransi nilai kategori baru (*unseen categories*) di seluruh encoder tanpa exception.
  - Toleransi nilai `null`/`NaN` pada inferensi langsung berkat automatic imputer step pada pipeline.
  - Deteksi delimiter fleksibel pada upload CSV (koma `,`, titik-koma `;`, tab `\t`).
  - Pre-flight check kolom target: Penolakan dataset klasifikasi dengan 1 kelas unik (*constant target*) atau target yang 100% kosong dengan HTTP 400.
  - Validasi jumlah baris minimal (minimal 5 baris) untuk mencegah kegagalan pembagian Cross-Validation.
- [x] Native Model & Data Drift Monitoring Engine (`packages/ml_engine/monitoring/drift_detector.py`, `apps/api/routers/monitoring.py`, `apps/api/db_models/orm.py`, `apps/web/src/pages/DriftMonitoringPage.tsx`):
  - Pencatatan log inferensi asinkron (`PredictionLog`) pada `POST /predict` dan `POST /predict/batch-csv` tanpa menambah latensi respons klien (zero latency overhead).
  - Algoritma uji statistik KS-Test (Kolmogorov-Smirnov) & Wasserstein Distance untuk continuous numerical features.
  - Algoritma Population Stability Index (PSI) untuk categorical features.
  - Agregasi kesehatan model: `HEALTHY` (Green, <50% drift), `WARNING` (Yellow, moderate shift), `DRIFT_DETECTED` (Red, >=50% features drift).
  - API Traffic Simulator (`POST /api/projects/{id}/drift/simulate`) untuk menyimulasikan data produksi 'Normal' vs 'Drifted' secara instan.
  - Neumorphic Drift Monitoring Dashboard di React frontend lengkap dengan banner status kesehatan, kartu metrik, tabel per-fitur, dan modal visualisasi kurva distribusi ganda (*Baseline vs Production*) via Recharts.
- [x] Task-Aware Recommendations & Continuous Target EDA Charting (`packages/ml_engine/recommendations/engine.py`, `apps/api/routers/recommendations.py`, `apps/web/src/pages/EDAPage.tsx`):
  - Target EDA Tab: Menampilkan Continuous Distribution Histogram & ringkasan statistik (Mean, Median, Std, Skewness, Kurtosis, Min/Max) alih-alih pie chart untuk target regresi (menghilangkan glik "rainbow UFO pie chart").
  - Menghapus anomali rekomendasi "Class Imbalance" pada task regresi.
  - Menambahkan rekomendasi perlakuan target regresi yang valid: `Target Skewness` (`log1p_transform` / `yeo_johnson_transform`) dan `Target Outliers` (`use_robust_models` / `clip_target_iqr`).
- [x] Toleransi Kolom Dropped pada Model Serving (`packages/ml_engine/serving/inference.py`, `apps/api/routers/serving.py`):
  - Skema inferensi live hanya mewajibkan fitur yang aktif digunakan transformer (bukan remainder/dropped columns seperti `date` atau `street`).
  - Kolom dropped otomatis diisi dengan `NaN` sebelum prediktor dieksekusi sehingga Scikit-Learn tidak menolak input.
- [x] Estimasi Keyakinan & Uncertainty Quantification pada Model Regresi (`packages/ml_engine/serving/inference.py`, `apps/api/routers/serving.py`, `apps/web/src/pages/ResultPage.tsx`, `apps/web/src/types/index.ts`):
  - Live Model Output menyajikan persentase keyakinan model regresi berbasis $R^2$ Score dari evaluasi Cross-Validation.
  - Kartu estimasi ketidakpastian: Margin of Error ($\pm\text{MAE}$) dan Rentang Estimasi (*Prediction Range* $[\text{pred} - \text{MAE}, \text{pred} + \text{MAE}]$) untuk mencegah salah paham bahwa prediksi bernilai 100% mutlak.
  - Otomatis menyertakan kolom `confidence`, `margin_of_error`, `interval_lower`, dan `interval_upper` pada ekspor file CSV hasil Batch Prediction.
- [x] **Data Leakage Detection End-to-End** (`packages/ml_engine/detection/leakage_detector.py`, `packages/ml_engine/recommendations/engine.py`, `packages/ml_engine/preprocessing/pipeline_builder.py`, `apps/web/src/pages/TreatmentPage.tsx`, `packages/ml_engine/assistant/explainer.py`):
  - **Identifier Detection**: Mendeteksi kolom ID/UUID/index baris (unique ratio ≥ 95% pada kolom integer/string). Kolom float kontinu *tidak* terklasifikasi sebagai identifier meskipun setiap nilainya unik.
  - **Target Proxy Detection (Numerik)**: Menghitung Pearson + Spearman (ambil max) untuk target kontinu, dan Point-Biserial untuk target biner. Threshold: |korelasi| ≥ 0.95.
  - **Target Proxy Detection (Kategorikal)**: Menghitung Cramér's V untuk pasangan kategorikal–kategorikal. Threshold: V ≥ 0.90.
  - **Severity Grading**: `critical` (≥ 0.98), `high` (≥ 0.95), `medium` (≥ threshold). Identifier selalu `critical`.
  - **Integrasi Pipeline & UI**: Keputusan `drop_column` dari kategori `data_leakage` langsung mengecualikan fitur bocor dari preprocessing scikit-learn pipeline, ditampilkan dengan badge bahaya merah (`badge-danger`) di Treatment UI, serta didukung edukasi interaktif AI Explainer.
  - **34 Test Cases Leakage & Pipeline Integration** terverifikasi 100% passed.
- [x] **API Contract Testing Suite** (`tests/test_api_contracts.py`):
  - 21 test cases memverifikasi kepatuhan status HTTP (`200`, `201`, `204`, `400`, `404`, `422`), validasi Pydantic schema, dan isolasi in-memory SQLite (`StaticPool`).
  - Menguji endpoint `/health`, `/api/projects`, `/api/projects/{id}/datasets`, `/api/datasets/{id}`, `/api/recommendations/decision`, `/api/projects/{id}/predict`, `/api/assistant/explain-recommendation`, dan `/api/projects/{id}/drift/summary`.
- [x] **AI Benchmark Summary & Metric Readability Overhaul** (`packages/ml_engine/assistant/explainer.py`, `apps/web/src/pages/ExperimentsPage.tsx`):
  - Format nama model profesional Title Case (mengeliminasi snake_case mentah).
  - Isolasi perbandingan runner-up eksklusif ke arsitektur algoritma yang berbeda (mencegah keanehan komparasi model identik dengan selisih 0.0000).
  - Format angka ribuan dengan koma (`RMSE: 218,589.8038`) dan penjelasan intuitif persentase variansi $R^2$.
  - Parser markdown frontend `renderMarkdownBold` untuk merender sintaks `**` menjadi teks tebal berbobot tinggi.
  - Penyelarasan format desimal metrik $R^2$ pada kartu Best Performing Model (`0.8273 ± 0.3092`) konsisten dengan tabel perbandingan.

- [x] **Distributed Asynchronous Job Workers** (`packages/ml_engine/jobs/`, `apps/api/routers/jobs.py`, `apps/api/db_models/orm.py`):
  - Database job queue model (`Job`) dengan prioritas, progress tracking (0–100%), retry count, status atomik (`pending`, `running`, `completed`, `failed`, `cancelled`).
  - `JobWorker` yang dapat berjalan sebagai background daemon thread dalam lifespan FastAPI atau standalone distributed process.
  - Registri task fleksibel (`@register_task`) dan endpoint API `/api/jobs/enqueue`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel`.
- [x] **Multi-Tenancy & Enterprise Security** (`apps/api/security.py`, `apps/api/routers/auth.py`, `apps/api/db_models/orm.py`):
  - Entitas `Tenant`, `User`, dan `ApiKey` terisolasi dengan hashing aman PBKDF2-HMAC-SHA256.
  - Autentikasi ganda via JWT Bearer Token (`PyJWT`) dan `X-API-Key`.
  - Role-Based Access Control (`admin`, `data_scientist`, `viewer`) dan guard isolasi proyek multi-tenant (`verify_project_access`).
- [x] **Cloud Object Storage Adapter** (`packages/ml_engine/storage/`, `apps/api/routers/storage.py`):
  - Abstraksi `StorageAdapter` dengan implementasi `LocalStorageAdapter` dan `S3StorageAdapter` (kompatibel AWS S3, MinIO, Ceph, Cloudflare R2).
  - Autentikasi AWS SigV4 terintegrasi murni menggunakan Python standard library `hmac` & `hashlib` tanpa dependensi wajib boto3.
  - Presigned URL generation dan endpoint `/api/storage/config` & `/api/storage/url`.
- [x] **Meta-Learning Recommendation Ranker** (`packages/ml_engine/recommendations/meta_ranker.py`, `apps/api/routers/meta_learning.py`):
  - `MetaFeatureExtractor` mengekstrak 13 meta-features dataset tabular (log baris/kolom, skewness, missing ratio, outlier ratio, imbalance ratio).
  - Surrogate meta-learning model (`RandomForestRegressor` / `Ridge`) memprediksi expected metric lift $\hat{\Delta}$.
  - Mekanisme cold-start fallback dan hybrid scoring yang memadukan skor heuristik dengan bobot meta-learning secara adaptif.



