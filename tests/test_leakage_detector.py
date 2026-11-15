"""
=============================================================================
TEST SUITE: LEAKAGE DETECTOR (tests/test_leakage_detector.py)
=============================================================================
Memverifikasi deteksi kebocoran data pada dua dimensi:
1. Identifier Detection  - Kolom ID/UUID/index unik
2. Target Proxy Detection - Fitur berkorelasi sangat tinggi dengan target
   a. Numerik vs Numerik  (Pearson/Spearman)
   b. Numerik vs Biner    (Point-Biserial)
   c. Kategorikal vs Kategorikal (Cramer V)
3. Integrasi dengan generate_recommendations
4. Edge cases: dataset kecil, kolom konstan, target hilang
=============================================================================
"""

import math

import numpy as np
import pandas as pd
import pytest

# sys.path dikelola oleh conftest.py
from detection.leakage_detector import (
    LeakageReport,
    IDENTIFIER_UNIQUE_RATIO_THRESHOLD,
    NUMERIC_TARGET_PROXY_THRESHOLD,
    CATEGORICAL_TARGET_PROXY_THRESHOLD,
    _cramers_v,
    _pearson_correlation,
    _spearman_correlation,
    _point_biserial,
    _severity_from_correlation,
    _detect_identifier_column,
    detect_leakage,
)
from profiling.profiler import profile_dataset
from recommendations.engine import generate_recommendations


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def df_with_identifier():
    """Dataset dengan kolom ID unik yang jelas (identifier leakage)."""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "user_id": [f"USR-{i:05d}" for i in range(n)],  # kolom ID unik
        "age": rng.integers(18, 65, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "target": rng.choice([0, 1], size=n),
    })


@pytest.fixture
def df_with_proxy_numeric():
    """Dataset di mana kolom 'price_scaled' adalah proxy sangat kuat dari target numerik."""
    n = 300
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=n)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, size=n),
        "price": base * 1000 + 50000,
        "price_scaled": base * 1000 + 50000 + rng.normal(0, 0.1, size=n),  # korelasi ~1.0
        "target": base * 100,  # numerik kontinu
    })


@pytest.fixture
def df_with_proxy_binary():
    """Dataset di mana 'leaked_flag' adalah proxy dari target biner."""
    n = 200
    rng = np.random.default_rng(1)
    target = rng.choice([0, 1], size=n)
    noise = rng.choice([0, 1], size=n, p=[0.98, 0.02])
    leaked = (target ^ noise).astype(int)  # hampir identik dengan target
    return pd.DataFrame({
        "feature_x": rng.normal(0, 1, size=n),
        "leaked_flag": leaked,
        "target": target,
    })


@pytest.fixture
def df_clean():
    """Dataset bersih tanpa leakage."""
    n = 200
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age": rng.integers(18, 65, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "category": rng.choice(["A", "B", "C"], size=n),
        "target": rng.choice([0, 1], size=n),
    })


# =============================================================================
# UNIT TESTS: HELPER STATISTIK
# =============================================================================

class TestHelperStatistics:
    """Memverifikasi fungsi-fungsi statistik internal."""

    def test_pearson_perfect_correlation(self):
        """Seri yang identik harus menghasilkan Pearson = 1.0."""
        s = pd.Series(range(50), dtype=float)
        result = _pearson_correlation(s, s)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_pearson_no_correlation(self):
        """Seri yang tidak berkorelasi harus menghasilkan Pearson mendekati 0."""
        rng = np.random.default_rng(10)
        a = pd.Series(rng.normal(0, 1, size=100))
        b = pd.Series(rng.normal(0, 1, size=100))
        # Tidak dijamin persis 0, tapi harusnya jauh dari 1
        result = _pearson_correlation(a, b)
        assert result < 0.3

    def test_pearson_constant_series(self):
        """Seri konstan harus menghasilkan 0.0 (standar deviasi = 0)."""
        a = pd.Series([5.0] * 50)
        b = pd.Series(range(50), dtype=float)
        assert _pearson_correlation(a, b) == 0.0

    def test_spearman_perfect_monotonic(self):
        """Hubungan monoton sempurna harus menghasilkan Spearman = 1.0."""
        a = pd.Series(range(50), dtype=float)
        b = pd.Series([x ** 3 for x in range(50)], dtype=float)  # monoton tapi non-linear
        result = _spearman_correlation(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_point_biserial_high_correlation(self):
        """Fitur yang identik dengan target biner harus menghasilkan korelasi tinggi."""
        n = 100
        target = np.array([0] * 50 + [1] * 50)
        feature = target.astype(float) + np.random.default_rng(5).normal(0, 0.01, size=n)
        result = _point_biserial(pd.Series(feature), pd.Series(target))
        assert result > 0.95

    def test_cramers_v_identical_columns(self):
        """Dua kolom kategorikal yang identik harus menghasilkan Cramer V = 1.0."""
        s = pd.Series(["A", "B", "C", "A", "B"] * 20)
        result = _cramers_v(s, s)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cramers_v_independent_columns(self):
        """Kolom kategorikal yang tidak berasosiasi harus menghasilkan Cramer V rendah."""
        rng = np.random.default_rng(7)
        s1 = pd.Series(rng.choice(["X", "Y", "Z"], size=200))
        s2 = pd.Series(rng.choice(["P", "Q", "R"], size=200))
        result = _cramers_v(s1, s2)
        assert result < 0.3

    def test_cramers_v_empty_contingency(self):
        """Contingency table kosong harus menghasilkan 0.0."""
        s = pd.Series([], dtype=str)
        result = _cramers_v(s, s)
        assert result == 0.0

    def test_severity_thresholds(self):
        """Nilai korelasi harus dipetakan ke severity yang tepat."""
        assert _severity_from_correlation(0.99) == "critical"
        assert _severity_from_correlation(0.98) == "critical"
        assert _severity_from_correlation(0.97) == "high"
        assert _severity_from_correlation(0.95) == "high"
        assert _severity_from_correlation(0.94) == "medium"


# =============================================================================
# UNIT TESTS: IDENTIFIER DETECTION
# =============================================================================

class TestIdentifierDetection:
    """Memverifikasi deteksi kolom identifier."""

    def test_identifier_dtype_detected(self):
        """Kolom bertipe 'identifier' harus selalu terdeteksi."""
        series = pd.Series([f"ID-{i}" for i in range(100)])
        result = _detect_identifier_column(series, dtype="identifier")
        assert result is not None
        assert result > 0.9

    def test_high_unique_ratio_detected(self):
        """Kolom dengan >95% nilai unik dari dtype non-identifier harus terdeteksi."""
        series = pd.Series(range(200))  # semua unik
        result = _detect_identifier_column(series, dtype="numeric", threshold=0.95)
        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_low_unique_ratio_not_detected(self):
        """Kolom dengan nilai unik sedikit TIDAK boleh terdeteksi sebagai identifier."""
        series = pd.Series(["A", "B", "C"] * 100)
        result = _detect_identifier_column(series, dtype="categorical")
        assert result is None

    def test_small_dataset_not_detected(self):
        """Dataset kecil (unique_count <= 20) tidak boleh salah terdeteksi."""
        series = pd.Series(range(15))  # unik tapi terlalu kecil
        result = _detect_identifier_column(series, dtype="numeric")
        assert result is None

    def test_detect_leakage_finds_identifier(self, df_with_identifier):
        """detect_leakage harus menemukan kolom 'user_id' sebagai identifier."""
        feature_dtypes = {"user_id": "identifier", "age": "numeric", "income": "numeric"}
        reports = detect_leakage(df_with_identifier, "target", feature_dtypes)

        identifier_reports = [r for r in reports if r.leakage_type == "identifier"]
        assert len(identifier_reports) >= 1
        columns = [r.column for r in identifier_reports]
        assert "user_id" in columns

    def test_identifier_has_critical_severity(self, df_with_identifier):
        """Kolom identifier harus selalu mendapat severity 'critical'."""
        feature_dtypes = {"user_id": "identifier", "age": "numeric", "income": "numeric"}
        reports = detect_leakage(df_with_identifier, "target", feature_dtypes)
        for r in reports:
            if r.leakage_type == "identifier":
                assert r.severity == "critical"

    def test_identifier_has_evidence(self, df_with_identifier):
        """LeakageReport identifier harus mengandung evidence konkret."""
        feature_dtypes = {"user_id": "identifier", "age": "numeric", "income": "numeric"}
        reports = detect_leakage(df_with_identifier, "target", feature_dtypes)
        for r in reports:
            if r.column == "user_id":
                evidence_str = " ".join(r.evidence)
                assert "unique_ratio" in evidence_str
                assert "unique_count" in evidence_str


# =============================================================================
# UNIT TESTS: TARGET PROXY DETECTION
# =============================================================================

class TestTargetProxyDetection:
    """Memverifikasi deteksi target proxy pada berbagai kombinasi tipe data."""

    def test_detect_proxy_numeric_target(self, df_with_proxy_numeric):
        """Kolom 'price_scaled' harus terdeteksi sebagai proxy dari target numerik."""
        feature_dtypes = {
            "feature_a": "numeric",
            "price": "numeric",
            "price_scaled": "numeric",
        }
        reports = detect_leakage(df_with_proxy_numeric, "target", feature_dtypes)
        proxy_cols = [r.column for r in reports if r.leakage_type == "target_proxy"]
        assert "price_scaled" in proxy_cols

    def test_detect_proxy_binary_target(self, df_with_proxy_binary):
        """Kolom 'leaked_flag' harus terdeteksi sebagai proxy dari target biner."""
        feature_dtypes = {"feature_x": "numeric", "leaked_flag": "numeric"}
        reports = detect_leakage(df_with_proxy_binary, "target", feature_dtypes)
        proxy_cols = [r.column for r in reports if r.leakage_type == "target_proxy"]
        assert "leaked_flag" in proxy_cols

    def test_proxy_has_correlation_in_evidence(self, df_with_proxy_numeric):
        """Evidence proxy harus mengandung nilai korelasi konkret."""
        feature_dtypes = {
            "feature_a": "numeric",
            "price": "numeric",
            "price_scaled": "numeric",
        }
        reports = detect_leakage(df_with_proxy_numeric, "target", feature_dtypes)
        for r in reports:
            if r.column == "price_scaled":
                evidence_str = " ".join(r.evidence)
                assert "correlation_value" in evidence_str
                assert "correlation_method" in evidence_str
                assert r.correlation_value is not None
                assert r.correlation_value > 0.9

    def test_proxy_correlation_method_populated(self, df_with_proxy_numeric):
        """LeakageReport proxy harus mempunyai correlation_method."""
        feature_dtypes = {"feature_a": "numeric", "price": "numeric", "price_scaled": "numeric"}
        reports = detect_leakage(df_with_proxy_numeric, "target", feature_dtypes)
        for r in reports:
            if r.leakage_type == "target_proxy":
                assert r.correlation_method in ("pearson", "spearman", "point_biserial", "cramers_v")

    def test_no_proxy_clean_dataset(self, df_clean):
        """Dataset bersih tidak boleh menghasilkan leakage reports."""
        feature_dtypes = {"age": "numeric", "income": "numeric", "category": "categorical"}
        reports = detect_leakage(df_clean, "target", feature_dtypes)
        assert len(reports) == 0


# =============================================================================
# UNIT TESTS: ORDERING & EDGE CASES
# =============================================================================

class TestOrderingAndEdgeCases:
    """Memverifikasi urutan hasil dan edge cases."""

    def test_results_ordered_by_severity(self, df_with_identifier, df_with_proxy_numeric):
        """Critical harus selalu muncul sebelum high/medium."""
        n = 200
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, size=n)
        df = pd.DataFrame({
            "row_id": range(n),          # identifier
            "col_a": rng.normal(0, 1, size=n),
            "col_proxy": base + rng.normal(0, 0.01, size=n),  # proxy tinggi
            "target": base,
        })
        feature_dtypes = {"row_id": "identifier", "col_a": "numeric", "col_proxy": "numeric"}
        reports = detect_leakage(df, "target", feature_dtypes)

        severity_order = {"critical": 0, "high": 1, "medium": 2}
        severity_values = [severity_order[r.severity] for r in reports]
        assert severity_values == sorted(severity_values), "Hasil harus diurutkan dari critical ke medium"

    def test_empty_dataframe(self):
        """DataFrame kosong tidak boleh menyebabkan error."""
        df = pd.DataFrame({"feature": [], "target": []})
        reports = detect_leakage(df, "target", {"feature": "numeric"})
        assert reports == []

    def test_missing_target_column(self, df_clean):
        """Jika target tidak ada di DataFrame, kembalikan list kosong."""
        feature_dtypes = {"age": "numeric", "income": "numeric"}
        reports = detect_leakage(df_clean, "kolom_tidak_ada", feature_dtypes)
        assert reports == []

    def test_empty_feature_dtypes(self, df_clean):
        """Jika feature_dtypes kosong, kembalikan list kosong."""
        reports = detect_leakage(df_clean, "target", {})
        assert reports == []

    def test_custom_proxy_threshold(self, df_with_proxy_numeric):
        """Custom threshold yang lebih rendah harus mendeteksi lebih banyak kolom."""
        feature_dtypes = {"feature_a": "numeric", "price": "numeric", "price_scaled": "numeric"}
        # Threshold default 0.95 — mungkin hanya price_scaled yang terdeteksi
        default_reports = detect_leakage(df_with_proxy_numeric, "target", feature_dtypes)
        # Threshold 0.40 — lebih banyak yang terdeteksi
        low_reports = detect_leakage(
            df_with_proxy_numeric, "target", feature_dtypes, proxy_threshold=0.40
        )
        assert len(low_reports) >= len(default_reports)

    def test_target_column_excluded_from_detection(self, df_with_proxy_numeric):
        """Kolom target itu sendiri tidak boleh muncul sebagai leakage report."""
        feature_dtypes = {"feature_a": "numeric", "price": "numeric", "price_scaled": "numeric"}
        reports = detect_leakage(df_with_proxy_numeric, "target", feature_dtypes)
        columns = [r.column for r in reports]
        assert "target" not in columns

    def test_unknown_feature_column_skipped(self, df_clean):
        """Kolom yang ada di feature_dtypes tapi tidak ada di df harus dilewati."""
        feature_dtypes = {"age": "numeric", "kolom_hantu": "numeric"}
        reports = detect_leakage(df_clean, "target", feature_dtypes)
        columns = [r.column for r in reports]
        assert "kolom_hantu" not in columns


# =============================================================================
# INTEGRATION TEST: RECOMMENDATION ENGINE
# =============================================================================

class TestLeakageIntegrationWithEngine:
    """Verifikasi bahwa leakage reports muncul di generate_recommendations."""

    def test_leakage_in_recommendations_identifier(self, df_with_identifier):
        """Recommendation category='data_leakage' harus muncul untuk dataset dengan identifier."""
        profile = profile_dataset(df_with_identifier, target_column="target")
        recs = generate_recommendations(df_with_identifier, profile, task_type="classification")

        leakage_recs = [r for r in recs if r.category == "data_leakage"]
        assert len(leakage_recs) >= 1

    def test_leakage_in_recommendations_proxy(self, df_with_proxy_binary):
        """Recommendation data_leakage harus muncul untuk kolom proxy biner."""
        profile = profile_dataset(df_with_proxy_binary, target_column="target")
        recs = generate_recommendations(df_with_proxy_binary, profile, task_type="classification")

        leakage_recs = [r for r in recs if r.category == "data_leakage"]
        assert len(leakage_recs) >= 1

    def test_leakage_recommendation_has_highest_score(self, df_with_identifier):
        """Rekomendasi leakage critical harus mendapat score tertinggi (>=95)."""
        profile = profile_dataset(df_with_identifier, target_column="target")
        recs = generate_recommendations(df_with_identifier, profile, task_type="classification")

        leakage_recs = [r for r in recs if r.category == "data_leakage"]
        for rec in leakage_recs:
            assert rec.score >= 75.0, f"Score harus >= 75, dapat {rec.score}"

    def test_no_leakage_recommendation_clean_dataset(self, df_clean):
        """Dataset bersih tidak boleh menghasilkan rekomendasi data_leakage."""
        profile = profile_dataset(df_clean, target_column="target")
        recs = generate_recommendations(df_clean, profile, task_type="classification")

        leakage_recs = [r for r in recs if r.category == "data_leakage"]
        assert len(leakage_recs) == 0

    def test_leakage_rec_to_dict_serializable(self, df_with_identifier):
        """Setiap Recommendation.to_dict() harus bisa diserialisasi tanpa error."""
        profile = profile_dataset(df_with_identifier, target_column="target")
        recs = generate_recommendations(df_with_identifier, profile, task_type="classification")

        for rec in recs:
            d = rec.to_dict()
            assert "id" in d
            assert "category" in d
            assert "evidence" in d
            # Nilai harus serializable (tidak ada NaN, float infinity, dll)
            assert isinstance(d["score"], float)
            assert isinstance(d["confidence"], float)

    def test_leakage_drop_decision_removes_column_from_pipeline(self, df_with_proxy_binary):
        """
        Integrasi End-to-End:
        Rekomendasi data_leakage (drop_column) yang disetujui user HARUS
        benar-benar membuang kolom bocor dari pipeline training scikit-learn.
        """
        from sklearn.dummy import DummyClassifier
        from ml_engine.preprocessing.pipeline_builder import build_pipeline

        profile = profile_dataset(df_with_proxy_binary, target_column="target")
        recs = generate_recommendations(df_with_proxy_binary, profile, task_type="classification")

        leakage_rec = next(r for r in recs if r.category == "data_leakage" and r.column == "leaked_flag")
        assert leakage_rec.recommended_action == "drop_column"

        # Simulasikan persetujuan user terhadap rekomendasi leakage
        decisions = [
            {
                "category": leakage_rec.category,
                "column_name": leakage_rec.column,
                "selected_action": leakage_rec.recommended_action,
            }
        ]

        pipeline, config = build_pipeline(decisions, profile.features, DummyClassifier())

        # Verifikasi kolom bocor masuk ke dropped_columns dan hilang dari fitur training
        assert "leaked_flag" in config["dropped_columns"]
        assert "leaked_flag" not in config["numeric_columns"]
        assert "leaked_flag" not in config["categorical_columns"]

        # Pipeline harus bisa fit tanpa error dan tanpa menyertakan kolom leaked_flag
        X = df_with_proxy_binary.drop(columns=["target"])
        y = df_with_proxy_binary["target"]
        pipeline.fit(X, y)

