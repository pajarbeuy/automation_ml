from __future__ import annotations

# =============================================================================
# MODUL: CUSTOM SKLEARN TRANSFORMERS
# =============================================================================
# Deskripsi: Transformer kustom yang kompatibel dengan Scikit-Learn (fit & transform)
#            digunakan dalam pipeline preprocessing AutoML untuk tugas spesifik
#            seperti pencaplokan outlier (Winsorization) dan pengkodean frekuensi.
# =============================================================================

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# 1. WINSORIZER (PENANGANAN OUTLIER DENGAN METODE IQR)
# =============================================================================

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizer: Membatasi (clip) nilai ekstrem (outlier) pada rentang [Q1 - factor*IQR, Q3 + factor*IQR].
    Prinsip: Batas dipelajari (fit) HANYA dari data training, lalu diterapkan (transform) ke data test/validasi.
    """

    def __init__(self, factor: float = 1.5):
        # Pengali IQR (default 1.5 untuk membatasi outlier lazim)
        self.factor = factor

    def fit(self, X, y=None):
        """
        Menghitung persentil Q1 (25%) dan Q3 (75%) serta batas bawah/atas per kolom numerik.
        """
        import pandas as pd

        self.lower_: list[float] = []
        self.upper_: list[float] = []

        if hasattr(X, "values"):
            arr = X.values
        else:
            arr = np.array(X)

        for i in range(arr.shape[1]):
            col = arr[:, i].astype(float)
            col_clean = col[~np.isnan(col)]
            q1, q3 = np.percentile(col_clean, [25, 75])
            iqr = q3 - q1
            self.lower_.append(q1 - self.factor * iqr)
            self.upper_.append(q3 + self.factor * iqr)
        return self

    def transform(self, X, y=None):
        """
        Memotong (clip) nilai-nilai yang melebihi batas bawah/atas yang dipelajari saat fit.
        """
        if hasattr(X, "values"):
            arr = X.values.copy().astype(float)
        else:
            arr = np.array(X, dtype=float).copy()

        for i in range(arr.shape[1]):
            arr[:, i] = np.clip(arr[:, i], self.lower_[i], self.upper_[i])
        return arr


# =============================================================================
# 2. FREQUENCY ENCODER (PENGKODEAN KATEGORI BERBASIS FREKUENSI KEMUNCULAN)
# =============================================================================

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    FrequencyEncoder: Mengubah nilai kategorikal menjadi rasio frekuensi kemunculannya pada dataset training.
    Kategori baru yang tidak pernah muncul di training set akan diberi nilai 0.0 secara otomatis.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Menghitung proporsi frekuensi setiap nilai unik per kolom kategorikal.
        """
        import pandas as pd

        self.freq_maps_: list[dict] = []
        if hasattr(X, "values"):
            arr = X.values
        else:
            arr = np.array(X, dtype=object)

        for i in range(arr.shape[1]):
            col = arr[:, i]
            total = len(col)
            unique, counts = np.unique(col[col != None], return_counts=True)  # noqa: E711
            freq = {str(u): c / total for u, c in zip(unique, counts)}
            self.freq_maps_.append(freq)
        return self

    def transform(self, X, y=None):
        """
        Memetakan setiap nilai kategorikal ke frekuensi rasio yang sudah dihitung.
        """
        if hasattr(X, "values"):
            arr = X.values
        else:
            arr = np.array(X, dtype=object)

        result = np.zeros((arr.shape[0], arr.shape[1]), dtype=float)
        for i in range(arr.shape[1]):
            freq_map = self.freq_maps_[i]
            for j, val in enumerate(arr[:, i]):
                result[j, i] = freq_map.get(str(val), 0.0)
        return result

