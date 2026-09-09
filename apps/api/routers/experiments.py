from __future__ import annotations

# =============================================================================
# ROUTER FASTAPI: EXPERIMENTS (PELATIHAN & EVALUASI MODEL ML)
# =============================================================================
# Deskripsi: Mengelola seluruh endpoint API terkait benchmarking eksperimen ML,
#            penjadwalan background task pelatihan, ekstraksi metrik CV,
#            ekspor skrip Python (.py), ekspor laporan HTML, dan tracking MLflow.
# =============================================================================

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from db_models import Artifact, Dataset, Experiment, Metric, WorkflowDecision


_PKG = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from models.registry import list_models, BASELINE_MODELS, BASELINE_REGRESSION_MODELS
from profiling.profiler import FeatureProfile
from experiments.runner import run_experiment
from schemas import ExperimentCreate, ExperimentOut, MetricOut, ArtifactOut
from .utils import features_from_profile_json

router = APIRouter(tags=["experiments"])


# =============================================================================
# BAGIAN 1: DAFTAR MODEL YANG TERSEDIA
# =============================================================================

@router.get("/models", response_model=list[str])
def get_available_models(task_type: str = "classification"):
    """Mengembalikan daftar semua algoritma model ML yang didukung oleh registry."""
    return list_models(task_type=task_type)


# =============================================================================
# BAGIAN 2: TRIGGER PELATIHAN EKSPERIMEN (POST /projects/{id}/experiments)
# =============================================================================

@router.post(
    "/projects/{project_id}/experiments",
    response_model=list[ExperimentOut],
    status_code=status.HTTP_201_CREATED,
)
def run_experiments(
    project_id: str,
    payload: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Mengirimkan permintaan eksperimen pelatihan model ML.
    Secara instan mengembalikan record berstatus 'pending', lalu mengeksekusi
    pelatihan secara asinkron di Background Tasks FastAPI.
    """
    from db_models import Project

    # 1. Validasi Keberadaan Project & Kolom Target Label
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")
    if not project.target_column:
        raise HTTPException(status_code=400, detail="Kolom target belum ditentukan")

    task_type = project.task_type or "classification"

    # 2. Ambil Dataset Terakhir yang Sudah Di-profiling
    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Belum ada profil dataset")

    # 3. Ambil Keputusan Preprocessing yang Disetujui (Accepted / Overridden)
    decisions = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.status.in_(["accepted", "overridden"]),
        )
        .all()
    )
    decisions_dicts = [
        {"category": d.category, "column_name": d.column_name, "selected_action": d.selected_action}
        for d in decisions
    ]
    features = features_from_profile_json(dataset.profile_json)

    # 4. Validasi Tipe Kolom Target (Deteksi Klasifikasi vs Regresi)
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    try:
        df_check = pd.read_csv(dataset.storage_uri)
        target_series = df_check[project.target_column]
        target_type = type_of_target(target_series.dropna())
        if task_type == "classification" and target_type == "continuous":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Kolom target '{project.target_column}' berisi nilai kontinu "
                    f"(tipe='{target_type}'). "
                    f"Project ini diset ke Klasifikasi. Ubah task_type ke Regresi atau pilih kolom target kategorikal."
                ),
            )
        if target_type == "unknown":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Kolom target '{project.target_column}' memiliki tipe tak dikenal ('{target_type}')."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # 5. Tentukan Model yang Akan Dijalankan (Sesuai Pilihan User)
    model_names = list(payload.model_names)
    if not model_names:
        model_names = list(BASELINE_REGRESSION_MODELS if task_type == "regression" else BASELINE_MODELS)

    created_experiments: list[Experiment] = []

    # 6. Buat Record 'pending' & Tambahkan ke Background Tasks
    for model_name in model_names:
        experiment_id = str(uuid.uuid4())

        exp = Experiment(
            id=experiment_id,
            project_id=project_id,
            dataset_id=dataset.id,
            config_json={
                "model_name": model_name,
                "cv_folds": payload.cv_folds,
                "random_seed": payload.random_seed,
                "enable_feature_engineering": payload.enable_feature_engineering,
                "is_tuned": payload.is_tuned,
                "model_params": payload.model_params,
            },
            model_name=model_name,
            random_seed=payload.random_seed,
            status="pending",
            started_at=datetime.utcnow(),
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        created_experiments.append(exp)

        background_tasks.add_task(
            _run_and_persist,
            experiment_id=experiment_id,
            project_id=project_id,
            dataset_id=dataset.id,
            dataset_path=dataset.storage_uri,
            target_column=project.target_column,
            model_name=model_name,
            decisions_dicts=decisions_dicts,
            features=features,
            storage_path=settings.storage_path,
            random_seed=payload.random_seed,
            cv_folds=payload.cv_folds,
            primary_metric=payload.primary_metric,
            enable_feature_engineering=payload.enable_feature_engineering,
            task_type=task_type,
            model_params=payload.model_params,
        )

    return created_experiments


# =============================================================================
# BAGIAN 3: WORKER BACKGROUND TASK PELATIHAN (PRIVATE HELPER)
# =============================================================================

def _run_and_persist(
    experiment_id: str,
    project_id: str,
    dataset_id: str,
    dataset_path: str,
    target_column: str,
    model_name: str,
    decisions_dicts: list[dict],
    features: list,
    storage_path: str,
    random_seed: int,
    cv_folds: int,
    primary_metric: str,
    enable_feature_engineering: bool = False,
    task_type: str = "classification",
    model_params: dict[str, Any] | None = None,
) -> None:
    """Fungsi pekerja latar belakang (Worker): Melatih 1 model dan menyimpan hasil ke SQLite & MLflow."""
    from database import SessionLocal

    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            return

        exp.status = "running"
        db.commit()

        # Eksekusi eksperimen lewat ML Engine Core
        record = run_experiment(
            experiment_id=experiment_id,
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            target_column=target_column,
            model_name=model_name,
            decisions=decisions_dicts,
            profile_features=features,
            storage_path=storage_path,
            random_seed=random_seed,
            cv_folds=cv_folds,
            primary_metric=primary_metric,
            enable_feature_engineering=enable_feature_engineering,
            task_type=task_type,
            model_params=model_params,
        )

        # Update status & simpan metrik ke database
        exp.status = record.status
        exp.error_message = record.error_message
        exp.training_time_seconds = record.training_time_seconds
        exp.pipeline_hash = record.pipeline_hash
        exp.finished_at = datetime.utcnow()
        exp.config_json = record.config_json

        for key, val in record.metrics_flat.items():
            db.add(Metric(experiment_id=experiment_id, name=key, value=val, split="cv"))

        for fold_data in record.fold_metrics:
            fold_num = fold_data.get("fold", 0)
            for metric_name, metric_val in fold_data.get("metrics", {}).items():
                db.add(Metric(experiment_id=experiment_id, name=metric_name,
                              value=metric_val, split="cv_fold", fold=fold_num))

        for art in record.artifacts:
            db.add(Artifact(experiment_id=experiment_id, type=art["type"],
                            storage_uri=art["path"], checksum=art.get("checksum")))

        db.commit()

        # Opsional: Log ke MLflow Server
        _try_log_mlflow(
            experiment_id=experiment_id,
            project_id=project_id,
            model_name=model_name,
            random_seed=random_seed,
            cv_folds=cv_folds,
            primary_metric=primary_metric,
            metrics_flat=record.metrics_flat,
            artifacts=record.artifacts,
            config=record.config_json or {},
        )
    except Exception as exc:
        try:
            exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.status = "failed"
                exp.error_message = f"BACKGROUND_ERROR: {exc}"
                exp.finished_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _try_log_mlflow(
    experiment_id: str,
    project_id: str,
    model_name: str,
    random_seed: int,
    cv_folds: int,
    primary_metric: str,
    metrics_flat: dict,
    artifacts: list[dict],
    config: dict,
) -> None:
    """Mengirim hasil eksperimen ke MLflow secara aman tanpa menghentikan sistem jika error."""
    try:
        import os
        import pickle
        import mlflow
        import mlflow.sklearn

        tracking_uri = settings.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            return

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"project_{project_id}")

        is_tuned = bool(config.get("is_tuned"))
        model_name_tag = model_name if model_name.endswith("_tune") else (f"{model_name}_tune" if is_tuned else model_name)

        with mlflow.start_run(run_name=f"{model_name_tag}_{experiment_id[:8]}"):
            mlflow.log_params({
                "model_name": model_name_tag,
                "is_tuned": is_tuned,
                "experiment_id": experiment_id,
                "random_seed": random_seed,
                "cv_folds": cv_folds,
                "primary_metric": primary_metric,
                **{f"cfg_{k}": str(v) for k, v in config.items() if not isinstance(v, dict)},
            })
            mlflow.log_metrics({k: v for k, v in metrics_flat.items() if isinstance(v, (int, float))})

            # Daftarkan model pipeline (.pkl) ke MLflow Model Registry & kolom Models
            for art in artifacts:
                p = Path(art.get("path", ""))
                if p.exists():
                    if art.get("type") == "pipeline" or str(p).endswith("pipeline.pkl"):
                        try:
                            with open(p, "rb") as f:
                                pipeline_obj = pickle.load(f)

                            clean_name = model_name_tag.replace(" ", "_").lower()
                            reg_name = f"ml_autonomus_{clean_name}"
                            try:
                                mlflow.sklearn.log_model(
                                    sk_model=pipeline_obj,
                                    artifact_path="model",
                                    registered_model_name=reg_name,
                                    serialization_format="pickle",
                                )
                            except Exception:
                                # Fallback jika store backend tidak mendukung model registry langsung
                                mlflow.sklearn.log_model(
                                    sk_model=pipeline_obj,
                                    artifact_path="model",
                                    serialization_format="pickle",
                                )
                        except Exception:
                            pass
                    # Log artifak mentah
                    mlflow.log_artifact(str(p))
    except Exception:
        pass


# =============================================================================
# BAGIAN 4: ENDPOINT QUERY EXPERIMENTS, METRICS, & ARTIFACTS
# =============================================================================

@router.get("/projects/{project_id}/experiments", response_model=list[ExperimentOut])
def list_experiments(project_id: str, db: Session = Depends(get_db)):
    """Mengambil seluruh eksperimen pada sebuah project."""
    exps = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
        .all()
    )
    return exps


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Mengambil detail 1 eksperimen spesifik."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")
    return exp


@router.get("/experiments/{experiment_id}/metrics", response_model=list[MetricOut])
def get_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """Mengambil daftar metrik hasil evaluasi eksperimen."""
    return db.query(Metric).filter(Metric.experiment_id == experiment_id).all()


@router.get("/experiments/{experiment_id}/artifacts", response_model=list[ArtifactOut])
def get_artifacts(experiment_id: str, db: Session = Depends(get_db)):
    """Mengambil daftar file artifak eksperimen."""
    return db.query(Artifact).filter(Artifact.experiment_id == experiment_id).all()


@router.get("/experiments/{experiment_id}/confusion-matrix")
def get_confusion_matrix(experiment_id: str, db: Session = Depends(get_db)):
    """Mengembalikan Confusion Matrix dalam format JSON."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")

    metrics_art = (
        db.query(Artifact)
        .filter(Artifact.experiment_id == experiment_id, Artifact.type == "metrics")
        .first()
    )
    if not metrics_art:
        raise HTTPException(status_code=404, detail="Artifak metrik tidak ditemukan")

    metrics_path = Path(metrics_art.storage_uri)
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="File metrik tidak ada di disk")

    with open(metrics_path) as f:
        data = json.load(f)

    return JSONResponse(content={"confusion_matrix": data.get("confusion_matrix", [])})


@router.get("/experiments/{experiment_id}/residuals")
def get_residuals(experiment_id: str, db: Session = Depends(get_db)):
    """Mengembalikan Data Residual Actual vs Predicted untuk regresi."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")

    metrics_art = (
        db.query(Artifact)
        .filter(Artifact.experiment_id == experiment_id, Artifact.type == "metrics")
        .first()
    )
    if not metrics_art:
        raise HTTPException(status_code=404, detail="Artifak metrik tidak ditemukan")

    metrics_path = Path(metrics_art.storage_uri)
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="File metrik tidak ada di disk")

    with open(metrics_path) as f:
        data = json.load(f)

    return JSONResponse(content={"residual_data": data.get("residual_data", [])})


@router.get("/experiments/{experiment_id}/download/{artifact_type}")
def download_artifact(experiment_id: str, artifact_type: str, db: Session = Depends(get_db)):
    """Mengunduh file artifak spesifik (misal: pipeline.pkl, config.json, metrics.json)."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")

    art = (
        db.query(Artifact)
        .filter(Artifact.experiment_id == experiment_id, Artifact.type == artifact_type)
        .first()
    )
    if not art:
        raise HTTPException(status_code=404, detail=f"Artifak '{artifact_type}' tidak ditemukan")

    file_path = Path(art.storage_uri)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan di disk")

    return FileResponse(path=str(file_path), filename=file_path.name, media_type="application/octet-stream")


# =============================================================================
# BAGIAN 5: EKSPOR KODE PYTHON & PEMILIHAN MODEL KANDIDAT
# =============================================================================

@router.get("/experiments/{experiment_id}/code")
def export_python_code(experiment_id: str, db: Session = Depends(get_db)):
    """Mengekspor Pipeline eksperimen menjadi skrip Python mandiri (.py)."""
    from db_models import Project

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Hanya eksperimen selesai yang bisa diekspor")

    project = db.query(Project).filter(Project.id == exp.project_id).first()
    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == exp.project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )

    from export.code_generator import generate_pipeline_script
    from models.registry import SKLEARN_CLASS_MAP

    model_class = SKLEARN_CLASS_MAP.get(exp.model_name, exp.model_name)
    script = generate_pipeline_script(
        config=exp.config_json or {},
        model_name=model_class,
        seed=exp.random_seed or 42,
        experiment_id=experiment_id,
        dataset_filename=dataset.filename if dataset else "data.csv",
        target_column=project.target_column if project else "target",
        task_type=project.task_type if project else None,
    )

    filename = f"pipeline_{exp.model_name}_{experiment_id[:8]}.py"
    return PlainTextResponse(
        content=script,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        media_type="text/x-python",
    )


@router.post("/experiments/{experiment_id}/select-candidate")
def select_candidate(experiment_id: str, db: Session = Depends(get_db)):
    """Tandai 1 eksperimen sebagai model kandidat terbaik untuk project ini."""
    from db_models import Project

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment tidak ditemukan")
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Hanya eksperimen selesai yang bisa dipilih")

    project = db.query(Project).filter(Project.id == exp.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    project.status = f"candidate:{experiment_id}"
    db.commit()

    return {"experiment_id": experiment_id, "project_id": exp.project_id, "message": "Model kandidat berhasil dipilih"}


# =============================================================================
# BAGIAN 6: EKSPOR LAPORAN HTML LENGKAP (DOWNLOAD REPORT)
# =============================================================================

@router.get("/projects/{project_id}/report")
def download_report(project_id: str, db: Session = Depends(get_db)):
    """Mengekspor dan mengunduh laporan HTML mandiri dari seluruh aktivitas project."""
    from datetime import timezone
    from db_models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset tidak ditemukan")

    experiments = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
        .all()
    )

    decisions = (
        db.query(WorkflowDecision)
        .filter(WorkflowDecision.project_id == project_id)
        .all()
    )

    # Ambil kandidat terpilih jika ada
    candidate_exp = None
    confusion_matrix = None
    if project.status and project.status.startswith("candidate:"):
        candidate_id = project.status.replace("candidate:", "")
        candidate_exp = db.query(Experiment).filter(Experiment.id == candidate_id).first()
        if candidate_exp:
            metrics_art = (
                db.query(Artifact)
                .filter(Artifact.experiment_id == candidate_id, Artifact.type == "metrics")
                .first()
            )
            if metrics_art and Path(metrics_art.storage_uri).exists():
                with open(metrics_art.storage_uri) as f:
                    confusion_matrix = json.load(f).get("confusion_matrix")

    def _exp_to_dict(e: Experiment) -> dict:
        metrics = db.query(Metric).filter(Metric.experiment_id == e.id, Metric.split == "cv").all()
        return {
            "model_name": e.model_name,
            "status": e.status,
            "training_time_seconds": e.training_time_seconds,
            "random_seed": e.random_seed,
            "error_message": e.error_message,
            "metrics": [{"name": m.name, "value": m.value, "split": m.split} for m in metrics],
        }

    decisions_dicts = [
        {"column_name": d.column_name, "category": d.category,
         "selected_action": d.selected_action, "status": d.status}
        for d in decisions
    ]

    from export.report_generator import generate_html_report

    candidate_dict = _exp_to_dict(candidate_exp) if candidate_exp else None
    if candidate_exp:
        candidate_dict["config_json"] = candidate_exp.config_json

    html = generate_html_report(
        project_name=project.name,
        project_id=project_id,
        target_column=project.target_column or "",
        dataset_filename=dataset.filename,
        dataset_rows=dataset.rows or 0,
        dataset_columns=dataset.columns or 0,
        profile=dataset.profile_json or {},
        decisions=decisions_dicts,
        experiments=[_exp_to_dict(e) for e in experiments],
        candidate_experiment=candidate_dict,
        confusion_matrix=confusion_matrix,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        task_type=project.task_type or "classification",
    )

    filename = f"automl_report_{project.name.replace(' ', '_')}_{project_id[:8]}.html"
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =============================================================================
# BAGIAN 7: HYPERPARAMETER TUNING ENGINE ENDPOINT
# =============================================================================

@router.post("/projects/{project_id}/experiments/tune")
def tune_hyperparameters(
    project_id: str,
    payload: dict,
    db: Session = Depends(get_db),
):
    """
    Menjalankan Hyperparameter Optimization (RandomizedSearch + Stratified/KFold CV)
    pada algoritma model ML yang dipilih.
    """
    import pandas as pd
    from db_models import Project
    from models.registry import get_estimator
    from preprocessing.pipeline_builder import build_pipeline
    from experiments.tuner import run_hyperparameter_search

    model_name = payload.get("model_name", "random_forest")
    n_trials = int(payload.get("n_trials", 10))
    random_seed = int(payload.get("random_seed", 42))
    cv_folds = int(payload.get("cv_folds", 5))
    enable_feature_engineering = bool(payload.get("enable_feature_engineering", False))

    # 1. Validasi Project & Dataset
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")
    if not project.target_column:
        raise HTTPException(status_code=400, detail="Target column belum ditentukan")

    task_type = project.task_type or "classification"
    default_metric = "r2" if task_type == "regression" else "f1_weighted"
    primary_metric = payload.get("primary_metric", default_metric)

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Dataset belum di-profiling")

    # 2. Muat Data CSV
    df = pd.read_csv(dataset.storage_uri)
    X = df.drop(columns=[project.target_column])
    y = df[project.target_column]

    # 3. Ambil Keputusan Preprocessing (Accepted / Overridden)
    decisions = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.status.in_(["accepted", "overridden"]),
        )
        .all()
    )
    decisions_dicts = [
        {"category": d.category, "column_name": d.column_name, "selected_action": d.selected_action}
        for d in decisions
    ]
    features = features_from_profile_json(dataset.profile_json)

    # 4. Bangun Pipeline dengan Estimator Model
    estimator = get_estimator(model_name, random_seed=random_seed, task_type=task_type)
    pipeline, _ = build_pipeline(
        decisions_dicts,
        features,
        estimator,
        enable_feature_engineering=enable_feature_engineering,
    )

    # 5. Jalankan Tuning Search
    result = run_hyperparameter_search(
        pipeline=pipeline,
        X=X,
        y=y,
        model_name=model_name,
        n_trials=n_trials,
        random_seed=random_seed,
        primary_metric=primary_metric,
        cv_folds=cv_folds,
        task_type=task_type,
    )

    # 6. Log ke MLflow Server jika tersedia
    _try_log_mlflow(
        experiment_id=str(uuid.uuid4()),
        project_id=project_id,
        model_name=f"{model_name}_tune",
        random_seed=random_seed,
        cv_folds=cv_folds,
        primary_metric=primary_metric,
        metrics_flat={f"{primary_metric}_tuned_score": result.best_score},
        artifacts=[],
        config={"is_tuned": True, "best_params": str(result.best_params), "trials_count": len(result.trials)},
    )

    return {
        "model_name": result.model_name,
        "best_params": result.best_params,
        "best_score": result.best_score,
        "primary_metric": result.primary_metric,
        "trials": result.trials,
        "total_time_seconds": result.total_time_seconds,
    }



