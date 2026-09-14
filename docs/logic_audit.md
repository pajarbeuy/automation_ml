# Audit Logika ML Engine — Temuan & Rekomendasi

> Dibuat: 2026-09-11 | Status: Diskusi

---

## Ringkasan

Dari 8 modul yang diaudit, ditemukan **7 keanehan logika** dengan tingkat keparahan berbeda-beda.
Tidak ada bug yang menyebabkan crash atau hasil salah secara hard (semuanya masih "bekerja"),
namun ada beberapa yang bisa menyebabkan **keputusan yang tidak akurat** atau **inkonsistensi**.

---

## 🔴 Kritis (Bisa Sebabkan Keputusan Salah)

### Anomali 1 — `outlier_detector.py`: Outlier rate dihitung ulang padahal sudah ada di profil

**File:** `detection/outlier_detector.py` L38–42  
**Masalah:**

```python
series = df[f.name].dropna()
q1, q3 = series.quantile(0.25), series.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
n_iqr = int(((series < lower) | (series > upper)).sum())
```

Kode ini menghitung ulang outlier rate IQR dari DataFrame, padahal:
- `f.outlier_rate_iqr` **sudah dihitung di profiler** dengan logika identik
- Hasilnya disimpan ke `n_outliers_iqr` (dari perhitungan baru)
- Tapi `outlier_rate_iqr` di `OutlierReport` diambil dari **`f.outlier_rate_iqr`** (dari profil)

Ini menyebabkan **inkonsistensi potensial** jika ada perbedaan implementasi antara keduanya.
Selain itu, IQR factor di outlier_detector hardcode `1.5`, sama dengan profiler — saat ini konsisten,
tapi rentan jika salah satu berubah.

**Rekomendasi:** Gunakan langsung `f.outlier_rate_iqr` dari profil, tidak perlu hitung ulang dari df.

---

### Anomali 2 — `recommendations/engine.py`: Scaling direction logic terbalik (Bug Logika)

**File:** `recommendations/engine.py` L346  
**Masalah:**

```python
elif skewness and abs(skewness) < 0.5:
    action = "standard_scaling"
    rationale = "Distribusi data relatif simetris..."
```

Kondisi `elif skewness and abs(skewness) < 0.5` menggunakan `skewness` sebagai truthiness check.
Jika `skewness == 0.0` (distribusi sempurna simetris), ekspresi `if skewness` adalah **False**,
sehingga kasus "distribusi sempurna simetris" justru jatuh ke `else` (StandardScaler default tanpa
rationale spesifik), bukan ke branch yang benar.

Ini minor tapi logikanya tidak tepat: skewness=0 adalah distribusi paling ideal untuk StandardScaler,
harusnya justru masuk ke branch ini.

**Rekomendasi:** Ubah ke `elif skewness is not None and abs(skewness) < 0.5:`

---

## 🟠 Sedang (Inkonsistensi Perilaku)

### Anomali 3 — `profiler.py`: `_detect_dtype` deteksi identifier string terlalu ketat

**File:** `profiling/profiler.py` L124–127  
**Masalah:**

```python
if series.nunique() == len(series.dropna()) and series.nunique() > 100:
    is_id_name = any(k in series.name.lower() for k in (...))
    if is_id_name:
        return "identifier"
```

Untuk string, profiler hanya menandai sebagai identifier jika **KEDUANYA**:
1. Semua nilai unik (`nunique() == len(dropna())`)
2. Nama kolom mengandung kata kunci ID

Sedangkan `leakage_detector.py` mendeteksi identifier jika:
1. `unique_ratio >= 0.95` (tidak harus 100% unik)

Ini berarti kolom string seperti `"transaction_token"` dengan 97% nilai unik:
- Profiler mendeteksinya sebagai **categorical** (karena `unique_ratio != 1.0`)
- Leakage detector mendeteksinya sebagai **identifier** (karena `ratio >= 0.95`)

**Inkonsistensi** ini tidak salah secara fatal, tapi membingungkan — profil mengatakan "categorical",
engine mengatakan "identifier".

**Rekomendasi:** Selaraskan threshold di profiler: jika `unique_ratio > 0.95 AND name contains ID keyword`, tandai sebagai identifier.

---

### Anomali 4 — `feature_engineering.py`: Interaction features bisa memperparah multikolinearitas

**File:** `preprocessing/feature_engineering.py` L62–70  
**Masalah:**

```python
corr_matrix = df_temp.corr().abs()
pairs.sort(key=lambda x: x[0], reverse=True)
self.interaction_pairs_idx_ = [(p[1], p[2]) for p in pairs[:self.max_interactions]]
```

Interaksi (`fitur_A * fitur_B`) dipilih berdasarkan **korelasi tertinggi** antar fitur — ini justru terbalik dari best practice. Fitur yang sudah sangat berkorelasi, jika dikalikan, menghasilkan kolom baru yang sangat redundan (multikolinearitas tinggi).

Contoh: Jika `income` dan `salary` berkorelasi 0.99, `income * salary` hampir identik dengan keduanya — tidak menambah informasi, malah menyebabkan masalah pada model berbasis gradient.

**Rekomendasi:** Pilih pasangan dengan korelasi **rendah-sedang** (0.3–0.7), bukan yang tertinggi. Atau filter: hanya buat interaksi jika korelasi antar-fitur < 0.8.

---

### Anomali 5 — `inference.py`: `interval_lower` bisa negatif untuk target positif

**File:** `serving/inference.py` L139  
**Masalah:**

```python
row_res["interval_lower"] = round(max(0.0, pred_val - mae_val), 2)
```

`max(0.0, ...)` hanya melindungi interval bawah dari negatif, tapi ini **hanya valid jika target selalu non-negatif** (misal: harga rumah). Untuk target yang bisa negatif (suhu, skor penilaian, profit/loss), ini memotong interval yang seharusnya valid.

Selain itu, `interval_upper` tidak memiliki proteksi serupa untuk domain tertentu (misal: probabilitas > 1.0, atau persentase > 100%).

**Rekomendasi:** Hapus `max(0.0, ...)` — biarkan interval bawah negatif jika memang domain mengizinkan. Serahkan validasi ke konteks domain user.

---

## 🟡 Minor (Desain Kurang Optimal)

### Anomali 6 — `charts.py` `target_vs_features`: Tidak skip kolom target dari fitur

**File:** `eda/charts.py` L142  
**Masalah:**

```python
for f in profile.features:
    if f.dtype == "numeric":
        ...
```

`profile.features` **sudah tidak menyertakan target** (difilter saat profiling di `profiler.py` L238–240).
Jadi ini bukan bug, tetapi tidak ada guard eksplisit jika caller meneruskan `profile.features` yang mengandung target.

Lebih penting: `target_vs_features` tidak memvalidasi apakah `target_col` ada di `df`.
Jika `target_col` tidak ada, line `target = df[target_col]` akan crash dengan `KeyError`.

**Rekomendasi:** Tambahkan guard: `if target_col not in df.columns: return {}`

---

### Anomali 7 — `FrequencyEncoder`: Nilai NaN dianggap sebagai kategori valid

**File:** `preprocessing/transformers.py` L94  
**Masalah:**

```python
unique, counts = np.unique(col[col != None], return_counts=True)  # noqa: E711
```

Filter `col != None` hanya menghilangkan Python `None`, **bukan** `np.nan`.
Jika kolom mengandung `np.nan`, nilai NaN dianggap kategori valid dan mendapat frekuensinya sendiri.
Saat transform, baris dengan NaN akan di-encode sebagai frekuensi NaN (bukan 0.0).

Ini berbeda dari perilaku imputer yang biasanya mengisi NaN sebelum masuk encoder,
tapi jika urutan step salah di pipeline, encoder bisa menerima NaN.

**Rekomendasi:** Ubah filter ke `col[~pd.isnull(col)]` atau `col[pd.notna(col)]` untuk menangkap semua varian missing (None, np.nan, pd.NA, pd.NaT).

---

## Ringkasan Prioritas Fix & Status Implementasi

| # | Severity | File | Deskripsi | Status | Dampak |
|---|----------|------|-----------|--------|--------|
| 1 | 🔴 | `outlier_detector.py` | Hitung ulang IQR duplikat | ✅ **Resolved** | Inkonsistensi teratasi, pakai nilai profil |
| 2 | 🔴 | `engine.py` | `skewness` truthiness bug | ✅ **Resolved** | `skewness=0.0` kini masuk standard scaling |
| 3 | 🟠 | `profiler.py` | Identifier threshold string beda dengan leakage_detector | ✅ **Resolved** | Threshold 95% diselaraskan dengan leakage detector |
| 4 | 🟠 | `feature_engineering.py` | Interaction dipilih dari korelasi tertinggi (terbalik) | ✅ **Resolved** | Pasangan interaksi dipilih dari korelasi 0.1–0.7 |
| 5 | 🟠 | `inference.py` | `interval_lower = max(0, ...)` terlalu asumtif | ✅ **Resolved** | Asumsi dihapus, mendukung target bernilai negatif |
| 6 | 🟡 | `charts.py` | Tidak guard KeyError pada target_col | ✅ **Resolved** | Guard `if not target_col or target_col not in df.columns` & skip self |
| 7 | 🟡 | `transformers.py` | `FrequencyEncoder` tidak filter np.nan | ✅ **Resolved** | Gunakan `pd.notna` saat fit & `pd.isna` saat transform (output 0.0) |

---

## Yang Sudah Benar ✅

- **`missing_detector.py`**: Sederhana dan benar — hanya delegasi dari profil, tidak ada redundansi.
- **`imbalance_detector.py`**: Logic guard `len(class_distribution) > 20` mencegah false positive pada regresi multi-class.
- **`leakage_detector.py`**: Sudah benar. Float exclusion dari identifier adalah keputusan yang tepat.
- **`pipeline_builder.py`**: Leakage-safe, `_clone_step` mencegah shared state antar pipeline.
- **`profiler.py`** (Bagian numerik): Robust, semua edge case ditangani dengan benar.
- **`inference.py`** (Detection bagian fitur): Logic deteksi required_features dari `transformers_` sudah tepat.
