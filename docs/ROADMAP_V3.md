# AutoML Platform — Product Roadmap V3

> **An evidence-driven, task-aware AutoML platform for dataset profiling, preprocessing recommendation, pipeline generation, experimentation, model serving, drift monitoring, AI-assisted workflow explanation, and autonomous NLP sentiment analysis.**

---

## 0. Product Principles

Throughout all development phases, the platform strictly adheres to six core principles:

1. **Evidence Before Recommendation**: Every preprocessing recommendation must be grounded in measurable data profiling signals (e.g. skewness, missing rate, lexical diversity, imbalance ratio).
2. **Task-Aware & Modality-Aware**: Strictly differentiates between `Classification`, `Regression`, and `Sentiment Analysis`. Techniques that do not apply to a specific task (e.g. SMOTE on regression, or numeric Winsorizing on free text) are never recommended.
3. **Model-Aware Processing**: Preprocessing decisions consider the downstream estimator family (e.g. TF-IDF sublinear scaling for Naive Bayes/LinearSVC; standard scaling for distance-based estimators).
4. **Leakage Safety**: All stateful transformations (imputation, scaling, encoding, TF-IDF vocabulary creation) must be fitted **strictly on training folds** inside scikit-learn `Pipeline` and `ColumnTransformer`.
5. **Candidate-Based Empirical Validation**: Evaluates multiple holistic candidate pipelines (`Recommended`, `Baseline`, `Alternative`) using fast True Paired Cross-Validation with Student's t CI before committing.
6. **Transparent Explainability**: Deterministic statistical engines make all decisions; AI/LLM layers solely provide human-readable narratives, word attribution, and contextual workflow guidance.

---

## 1. Status of Implementation (Phases 0 — 8 Completed)

| Phase | Description | Status | Key Highlights |
|---|---|---|---|
| **Phase 0** | **Foundation** | ✅ Completed | Monorepo architecture, FastAPI backend, React 18 + Vite frontend, SQLite/PostgreSQL ORM, CSV file ingestion |
| **Phase 1** | **Dataset Profiling + EDA** | ✅ Completed | Automated schema detection, missing rates, skewness, kurtosis, duplicate detection, Pearson heatmap, distribution charts |
| **Phase 2** | **Recommendation Engine** | ✅ Completed | Rule-based engine, target proxy & identifier leakage detection, candidate generation, 3-fold empirical lift benchmark |
| **Phase 3** | **Pipeline Engine** | ✅ Completed | `ColumnTransformer` builder, custom `Winsorizer` & `FrequencyEncoder`, 16-char SHA256 pipeline hash, dry-run validator |
| **Phase 4** | **Experiment Engine** | ✅ Completed | Stratified K-Fold / K-Fold CV, baseline model tournament, metric cards (F1, Accuracy, ROC-AUC, PR-AUC), error surfacing |
| **Phase 5** | **Export + Tracking** | ✅ Completed | `.pkl` pipeline export, config JSON, standalone reproducible `.py` code generator, HTML executive report, MLflow auto-logging |
| **Phase 6** | **AI Assistant** | ✅ Completed | Rule-based & LLM recommendation explainer, experiment benchmark summarizer, guided workflow navigation widget |
| **Phase 7** | **Advanced AutoML** | ✅ Completed | Regression support ($R^2$, MAE, MSE, RMSE, residual plots), `AutoFeatureEngineer` (log1p & interaction generator), `RandomizedSearchCV` tuning |
| **Phase 8** | **Enterprise & Production** | ✅ Completed | Model serving REST API, Neumorphic Inference Playground, native KS-Test & PSI drift monitoring, WSL Docker compose, **Multi-Tenancy & RBAC**, **Cloud Object Storage (S3/MinIO SigV4)**, **Distributed Asynchronous Job Workers**, **Meta-Learning Recommendation Ranker**, **Frontend Auth Guard** |
| **Phase 9** | **Autonomous NLP Sentiment Engine** | ✅ Completed | Lexical profiler, `TextCleanerTransformer`, zero-leakage TF-IDF pipelines, NLP tournament (NB, LogReg, LinearSVC, SGD), word attribution explainability, text EDA tabs, live sentiment playground |

---

## 2. Phase 9 — Autonomous Sentiment Analysis Engine (Completed)

### Goal
Provide end-to-end autonomous text sentiment analysis for tabular datasets containing text reviews, customer feedback, and sentiment polarity targets, while guaranteeing zero data leakage and computational efficiency.

### 9.1 Text Profiling & Lexical Diagnostics
- [x] Schema detection identifying high-cardinality natural language text columns (`dtype="text"`).
- [x] Lexical diversity calculation: Type-Token Ratio (TTR), total vocabulary size, average document character length, token count per row.
- [x] Linguistic noise detection: stopword ratio, uppercase shouting percentage, punctuation density, emoji frequency.
- [x] Class distribution and sentiment imbalance detection (`positive`, `negative`, `neutral` or binary).
- [x] Text EDA visualizer: token length distribution histogram, top-N unigram/bigram frequency tables.

### 9.2 Custom Text Preprocessing & Leakage-Safe Pipelines
- [x] `TextCleanerTransformer`: Lowercasing, URL/HTML stripping, punctuation normalization, and whitespace cleanup in scikit-learn format.
- [x] `TfidfVectorizer` / `CountVectorizer` integration into `ColumnTransformer`:
  - Vocabulary fitting strictly within cross-validation training folds.
  - Zero leakage across test/holdout folds; graceful handling of Out-Of-Vocabulary (OOV) tokens.
- [x] N-gram candidate configurations: `(1, 1)` unigrams, `(1, 2)` unigram + bigram combinations, and character n-grams `(3, 5)` for typo resilience.
- [x] Negation-preserving stopword tokenization (retaining negative indicators such as `not`, `never`, `tidak`, `bukan`).

### 9.3 NLP Preprocessing Recommendation & Candidate Strategy
- [ ] Evidence-based text recommendation rules:
  - If vocabulary is large and documents are long $\rightarrow$ recommend sublinear TF-IDF scaling + unigram/bigram `(1, 2)` + `min_df=3`.
  - If documents are short and slang-heavy $\rightarrow$ recommend character n-grams and stopword retention.
- [ ] Candidate Strategy Generation:
  - `Recommended`: Full clean + sublinear TF-IDF + bigram + min_df filtering.
  - `Baseline`: Basic CountVectorizer (unigrams only) + simple lowercasing.
  - `Alternative`: Typo-tolerant sublinear TF-IDF with character n-grams `(3, 5)`.
- [ ] Fast 3-Fold Paired Cross-Validation candidate screening to evaluate empirical lift before training full ensembles.

### 9.4 NLP Estimator Tournament
- [x] Estimator registry for text classification:
  - `MultinomialNB` (Fast probabilistic baseline)
  - `ComplementNB` (Optimized for imbalanced text classification)
  - `LogisticRegression` (L2 / ElasticNet regularized with sparse TF-IDF)
  - `LinearSVC` (High-performance sparse linear margin classifier)
  - `SGDClassifier` (Scalable linear classifier with log loss / modified Huber)
- [x] Stratified Cross-Validation reporting F1-Macro, Precision, Recall, Accuracy, and PR-AUC.
- [x] Primary metric configuration with pre-training commitment (default: `f1_macro`).

### 9.5 Sentiment Explainability & Word Attribution
- [x] Linear coefficient & log-probability extractor:
  - Surface top-15 positive words and top-15 negative words driving model predictions.
- [x] Visual Word Attribution Card:
  - Interactive highlight badges in Result Page displaying which tokens contributed most heavily to the predicted sentiment score.
- [x] Standalone reproducible Python script generator (`code_generator.py`) for the entire NLP text pipeline.

### 9.6 Live Text Inference & Serving
- [x] REST API endpoint for raw text prediction (`POST /api/projects/{id}/predict`):
  - Input: raw string text (no manual tokenization needed by client).
  - Output: predicted sentiment (`positive` / `negative` / `neutral`), confidence score (0.0 to 1.0), and top contributing word highlights.
- [x] Live Inference Playground UI tab for interactive sentence testing.
- [x] Batch CSV text prediction with automated sentiment annotation and download.
### 9.7 Text & Vocabulary Drift Monitoring
- [ ] Asynchronous prediction logging for incoming inference text payloads.
- [ ] Out-Of-Vocabulary (OOV) rate calculation: percentage of tokens in production requests not seen during model training.
- [ ] Document token length distribution drift (KS-Test on production vs baseline token lengths).
- [ ] Sentiment distribution shift monitoring (PSI on predicted class proportions over time).

---

## 3. Phase 10 — Enterprise Scale & Governance (Future Horizon)

### Goal
Scale the platform for high-throughput multi-node enterprise deployments with multi-modal fusion and compliance auditing.

### 10.1 Multi-Modal Fusion (Tabular + Text)
- [ ] Hybrid `ColumnTransformer` joining numerical metadata (e.g. product price, customer tenure), categorical features, and text review tokens into a unified model feature vector.
- [ ] Multi-modal candidate preprocessing generation.

### 10.2 Distributed Multi-Node Worker Cluster
- [ ] Redis / RabbitMQ broker integration for `JobWorker`.
- [ ] Distributed Celery workers executing across multiple Docker nodes or Kubernetes pods.

### 10.3 Enterprise Model Governance & Compliance
- [ ] Comprehensive audit log tracking all user workflow decisions (`who approved what transformation and when`).
- [ ] Model card generation (HTML / PDF) documenting data lineage, performance across demographic subsets, and limitation disclosures.
- [ ] Automated model retirement and rollback policies based on drift alert triggers.
