from __future__ import annotations

# =============================================================================
# MODUL: PYTHON CODE GENERATOR (PEMBUAT KODE PYTHON MANDIRI)
# =============================================================================
# Deskripsi: Mengubah konfigurasi eksperimen (config_json) menjadi skrip Python murni
#            yang dapat dijalankan secara independen di luar platform AutoML
#            (misalnya di Jupyter Notebook atau sistem produksi).
#
# Keunggulan Skrip yang Dihasilkan:
# - Mengonstruksi ulang Scikit-Learn Pipeline persis seperti eksperimen.
# - Dilengkapi contoh lengkap fit(), predict(), cross-validation, dan ekspor .pkl.
# - Reproduksibel 100% (menggunakan random seed yang sama).
# - Tanpa ketergantungan (zero dependency) pada codebase AutoML ini.
# =============================================================================

import textwrap
from typing import Any


# =============================================================================
# BAGIAN 1: PEMBANTU PEMETAAN AKSI PREPROCESSING
# =============================================================================

def _imputer_code(action: str) -> str:
    """Mengembalikan string kode perakitan SimpleImputer / KNNImputer."""
    map_: dict[str, str] = {
        "mean_imputation":          "SimpleImputer(strategy='mean')",
        "median_imputation":        "SimpleImputer(strategy='median')",
        "most_frequent_imputation": "SimpleImputer(strategy='most_frequent')",
        "constant_imputation":      "SimpleImputer(strategy='constant', fill_value='missing')",
        "knn_imputation":           "KNNImputer(n_neighbors=5)",
    }
    return map_.get(action, "SimpleImputer(strategy='median')")


def _scaler_code(action: str) -> str | None:
    """Mengembalikan string kode Feature Scaler."""
    map_: dict[str, str | None] = {
        "standard_scaling":  "StandardScaler()",
        "min_max_scaling":   "MinMaxScaler()",
        "robust_scaling":    "RobustScaler()",
        "no_scaling":        None,
        "leave_unchanged":   None,
    }
    return map_.get(action)


def _encoder_code(action: str) -> str:
    """Mengembalikan string kode Categorical Encoder."""
    map_: dict[str, str] = {
        "one_hot_encoding":   "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
        "ordinal_encoding":   "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
        "frequency_encoding": "FrequencyEncoder()",
    }
    return map_.get(action, "OneHotEncoder(handle_unknown='ignore', sparse_output=False)")


def _needs_frequency_encoder(config: dict) -> bool:
    """Mengecek apakah pipeline membutuhkan class FrequencyEncoder kustom."""
    for c in config.get("categorical_config", {}).values():
        if c.get("encoding") == "frequency_encoding":
            return True
    return False


def _needs_winsorizer(config: dict) -> bool:
    """Mengecek apakah pipeline membutuhkan class Winsorizer kustom."""
    for c in config.get("numeric_config", {}).values():
        if c.get("outlier") == "winsorization":
            return True
    return False


# =============================================================================
# BAGIAN 2: KAMUS IMPORT & KONSTRUKTOR MODEL ML
# =============================================================================

_MODEL_IMPORTS: dict[str, str] = {
    "DummyClassifier":          "from sklearn.dummy import DummyClassifier",
    "LogisticRegression":       "from sklearn.linear_model import LogisticRegression",
    "LinearSVC":                "from sklearn.svm import LinearSVC",
    "RandomForestClassifier":   "from sklearn.ensemble import RandomForestClassifier",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "HistGradientBoostingClassifier": "from sklearn.ensemble import HistGradientBoostingClassifier",
    "KNeighborsClassifier":     "from sklearn.neighbors import KNeighborsClassifier",
}

_MODEL_CONSTRUCTORS: dict[str, str] = {
    "DummyClassifier":           "DummyClassifier(strategy='most_frequent')",
    "LogisticRegression":        "LogisticRegression(max_iter=1000, random_state=SEED)",
    "LinearSVC":                 "LinearSVC(max_iter=2000, random_state=SEED)",
    "RandomForestClassifier":    "RandomForestClassifier(n_estimators=100, random_state=SEED)",
    "GradientBoostingClassifier": "GradientBoostingClassifier(n_estimators=100, random_state=SEED)",
    "HistGradientBoostingClassifier": "HistGradientBoostingClassifier(random_state=SEED)",
    "KNeighborsClassifier":      "KNeighborsClassifier(n_neighbors=5)",
}


# =============================================================================
# BAGIAN 3: GENERATOR KODE PYTHON UTAMA (GENERATE_PIPELINE_SCRIPT)
# =============================================================================

def generate_pipeline_script(
    config: dict[str, Any],
    model_name: str,
    seed: int = 42,
    experiment_id: str = "",
    dataset_filename: str = "data.csv",
    target_column: str = "target",
) -> str:
    """
    Menghasilkan skrip Python mandiri yang mereproduksi Pipeline eksperimen.

    Parameter:
        config: Dict config_json hasil eksperimen.
        model_name: Nama class Scikit-Learn model (misal: 'RandomForestClassifier').
        seed: Random seed eksperimen.
        experiment_id: ID Eksperimen (untuk header skrip).
        dataset_filename: Nama file dataset CSV untuk contoh kode.
        target_column: Nama kolom target label.

    Return:
        Kode skrip Python lengkap sebagai string.
    """
    numeric_config: dict[str, dict] = config.get("numeric_config", {})
    categorical_config: dict[str, dict] = config.get("categorical_config", {})
    dropped: list[str] = config.get("dropped_columns", [])

    needs_freq = _needs_frequency_encoder(config)
    needs_wins = _needs_winsorizer(config)

    model_import = _MODEL_IMPORTS.get(model_name, f"# Unknown model: {model_name}")
    model_ctor = _MODEL_CONSTRUCTORS.get(model_name, f"{model_name}()")

    # 1. Rakit kode transformer per kolom numerik
    transformer_lines = []

    for col, c in numeric_config.items():
        steps = []
        imputer = c.get("missing", "median_imputation")
        outlier = c.get("outlier", "leave_unchanged")
        scaling = c.get("scaling", "standard_scaling")

        if imputer and imputer != "drop_column":
            steps.append(f"('imputer', {_imputer_code(imputer)})")
        if outlier == "winsorization":
            steps.append("('winsorizer', Winsorizer())")
        sc = _scaler_code(scaling)
        if sc:
            steps.append(f"('scaler', {sc})")

        if steps:
            steps_str = ", ".join(steps)
            transformer_lines.append(
                f"    (\"num_{col}\", Pipeline([{steps_str}]), [\"{col}\"]),"
            )
        else:
            transformer_lines.append(f"    (\"num_{col}\", \"passthrough\", [\"{col}\"]),")

    # 2. Rakit kode transformer per kolom kategorikal
    for col, c in categorical_config.items():
        steps = []
        imputer = c.get("missing", "most_frequent_imputation")
        encoding = c.get("encoding", "one_hot_encoding")

        if imputer and imputer != "drop_column":
            steps.append(f"('imputer', {_imputer_code(imputer)})")
        steps.append(f"('encoder', {_encoder_code(encoding)})")

        steps_str = ", ".join(steps)
        transformer_lines.append(
            f"    (\"cat_{col}\", Pipeline([{steps_str}]), [\"{col}\"]),"
        )

    transformers_block = "\n".join(transformer_lines) if transformer_lines else "    # No feature transformers configured"

    # 3. Susun daftar import pustaka Python
    base_imports = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer, KNNImputer",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
        "from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder",
        model_import,
    ]
    if needs_freq or needs_wins:
        base_imports.append(
            "# Custom Transformers (FrequencyEncoder / Winsorizer)"
        )
        if needs_freq:
            base_imports.append(_FREQUENCY_ENCODER_INLINE)
        if needs_wins:
            base_imports.append(_WINSORIZER_INLINE)

    imports_block = "\n".join(base_imports)

    dropped_comment = ""
    if dropped:
        dropped_list = ", ".join(f'"{d}"' for d in dropped)
        dropped_comment = f"\n# CATATAN: Kolom-kolom berikut dibuang (dropped) oleh keputusan preprocessing:\n# DROPPED = [{dropped_list}]"

    # 4. Susun skrip lengkap menggunakan string template textwrap
    script = textwrap.dedent(f"""\
        # =============================================================================
        # SKRIP PYTHON HASIL EKSPOR AUTOML (REPRODUCIBLE ML PIPELINE)
        # ID Eksperimen : {experiment_id}
        # Model ML      : {model_name}
        # Random Seed   : {seed}
        # Dibuat oleh   : ml-autonomus AutoML System
        # =============================================================================
        # Skrip ini bersifat mandiri (self-contained).
        # Anda dapat menjalankan skrip ini di lingkungan Python mana pun
        # yang memiliki pustaka scikit-learn dan pandas terinstall.
        # =============================================================================

        {imports_block}
        {dropped_comment}

        SEED = {seed}
        TARGET = "{target_column}"


        # ---------------------------------------------------------------------------
        # DEFINISI PIPELINE SCIKIT-LEARN
        # ---------------------------------------------------------------------------

        preprocessor = ColumnTransformer(
            transformers=[
        {transformers_block}
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", {model_ctor}),
        ])


        # ---------------------------------------------------------------------------
        # CONTOH PENGGUNAAN (TRAINING, EVALUASI, & SIMPAN MODEL)
        # ---------------------------------------------------------------------------

        if __name__ == "__main__":
            # 1. Memuat Dataset
            df = pd.read_csv("{dataset_filename}")

            X = df.drop(columns=[TARGET])
            y = df[TARGET]

            # 2. Pembagian Data Train / Test (Split)
            from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )

            # 3. Fitting Pipeline pada Data Training
            pipeline.fit(X_train, y_train)

            # 4. Evaluasi Performa pada Data Test
            from sklearn.metrics import classification_report
            y_pred = pipeline.predict(X_test)
            print("=== Laporan Klasifikasi ===")
            print(classification_report(y_test, y_pred))

            # 5. Cross-Validation (Persis Seperti Eksperimen AutoML)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=["accuracy", "f1_weighted", "roc_auc_ovr_weighted"],
                return_train_score=False,
            )
            print("\n=== Hasil Cross Validation ===")
            for metric, vals in scores.items():
                if metric.startswith("test_"):
                    print(f"{{metric}}: {{vals.mean():.4f}} ± {{vals.std():.4f}}")

            # 6. Menyimpan Objek Pipeline yang Sudah Di-fit ke File Pickle (.pkl)
            import pickle
            with open("pipeline_export.pkl", "wb") as f:
                pickle.dump(pipeline, f)
            print("\nPipeline berhasil disimpan ke file 'pipeline_export.pkl'")
    """)

    return script


# =============================================================================
# BAGIAN 4: STUB KODE TRANSFORMER KUSTOM INLINE
# =============================================================================

_FREQUENCY_ENCODER_INLINE = '''
class FrequencyEncoder:
    """Mengkodekan nilai kategorikal berdasarkan frekuensi kemunculannya pada data train."""
    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else X[:, 0]
        counts = {}
        for v in col:
            counts[v] = counts.get(v, 0) + 1
        total = len(col)
        self.freq_map_ = {k: v / total for k, v in counts.items()}
        return self

    def transform(self, X):
        import numpy as np
        col = X.iloc[:, 0] if hasattr(X, "iloc") else X[:, 0]
        return np.array([self.freq_map_.get(v, 0.0) for v in col]).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
'''

_WINSORIZER_INLINE = '''
class Winsorizer:
    """Membatasi nilai ekstrem (outlier) berdasarkan rentang Interquartile Range (IQR)."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        import numpy as np
        data = X.ravel() if hasattr(X, "ravel") else np.array(X).ravel()
        q1, q3 = np.nanpercentile(data, [25, 75])
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        import numpy as np
        data = np.array(X, dtype=float)
        return np.clip(data, self.lower_, self.upper_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
'''

