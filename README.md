# Prediksi Harga Kendaraan Bekas Menggunakan Linear Regression & Gradient Boosting

## Deskripsi Proyek

Proyek ini merupakan bagian dari tugas mata kuliah Machine Learning yang bertujuan untuk membangun model *Machine Learning* untuk memprediksi harga jual kendaraan bekas (*Selling Price*). Prediksi dilakukan berdasarkan berbagai faktor seperti usia kendaraan, jarak tempuh (*kilometer*), jenis bahan bakar, tipe penjual, jenis transmisi, serta riwayat kepemilikan.

Proyek ini memanfaatkan Python dan Jupyter Notebook untuk melakukan seluruh tahapan *data science workflow*, mulai dari pembersihan data (*data cleaning*), rekayasa fitur (*feature engineering*), transformasi data, pelatihan model regresi, evaluasi performa, hingga penyimpanan model terbaik ke dalam file instansiasi (`.joblib`).

---

## Tujuan Proyek

1. **Pembersihan Data & Pencegahan Leakage:** Memeriksa dan menangani data duplikat eksak untuk mencegah *Data Leakage* antara data latih (*training*) dan data uji (*testing*).
2. **Rekayasa Fitur (Feature Engineering):** Mengekstrak fitur baru seperti merek kendaraan (`brand`) dan menghitung umur mobil (`car_age`) untuk meningkatkan daya prediksi model.
3. **Preprocessing & Skaling:** Mengatasi data miring (*skewed distribution*) dengan transformasi logaritmik (`np.log1p`) pada target dan fitur numerik, serta menerapkan `StandardScaler` secara akurat pada data latih.
4. **Evaluasi & Perbandingan Model:** Membandingkan performa algoritma **Linear Regression** dan **Gradient Boosting Regressor** berdasarkan metrik MAE, MSE, dan $R^2$ Score.
5. **Model Deployment Readiness:** Menyimpan model teruji dengan performa terbaik ke dalam file `.joblib` agar siap diintegrasikan ke aplikasi.

---

## Dataset

Dataset yang digunakan berisi informasi detail spesifikasi dan riwayat kendaraan bekas.

* **Ukuran Dataset Awal:** 4.340 baris $\times$ 8 kolom
* **Ukuran Dataset Setelah Cleaning:** 3.577 baris $\times$ 8 kolom (setelah eliminasi 763 baris duplikat eksak)

### Fitur Dataset Awal:
- `name` — Nama/model lengkap kendaraan (contoh: *Maruti 800 AC*, *Hyundai Verna 1.6 SX*).
- `year` — Tahun pembuatan kendaraan.
- `selling_price` — Harga jual kendaraan (Target Regresi).
- `km_driven` — Total jarak tempuh kendaraan dalam kilometer.
- `fuel` — Jenis bahan bakar (*Petrol*, *Diesel*, *Electric*, *LPG*, *CNG*).
- `seller_type` — Tipe penjual (*Individual*, *Dealer*, *Trustmark Dealer*).
- `transmission` — Tipe transmisi (*Manual*, *Automatic*).
- `owner` — Riwayat kepemilikan (*First Owner*, *Second Owner*, *Third Owner*, dst.).

---

## Tahapan & Preprocessing Data

1. **Data Cleaning:** Menghapus 763 baris duplikat eksak demi menjaga integritas pembagian dataset.
2. **Feature Extraction:**
   - `brand`: Mengekstrak kata pertama dari kolom `name` untuk mengelompokkan pabrikan mobil.
   - `car_age`: Menghitung umur kendaraan berdasarkan acuan tahun maksimum di dataset ($2020 - \text{year}$).
3. **Encoding Fitur Kategorikal:**
   - **Ordinal Encoding:** Diterapkan pada kolom `owner` karena memiliki hierarki tingkat kepemilikan.
   - **One-Hot Encoding:** Diterapkan pada kolom nominal (`brand`, `fuel`, `seller_type`, `transmission`) menggunakan `pd.get_dummies(drop_first=True)`.
4. **Data Splitting:** Membagi dataset menjadi 80% Data Latih (2.861 sampel) dan 20% Data Uji (716 sampel) menggunakan `train_test_split(random_state=1)`.
5. **Transformasi & Skaling:**
   - Penerapan logaritmik `np.log1p` pada target `selling_price` dan fitur `km_driven` untuk menstabilkan varians.
   - Standardisasi fitur numerik (`car_age`, `km_driven`, `owner_encoded`) menggunakan `StandardScaler` (proses `.fit()` hanya dilakukan pada data latih).

---

## Performa & Hasil Model

Model dievaluasi setelah mengembalikan nilai prediksi dari skala logaritma ke skala nominal asli (Rupiah/Mata Uang Asli) menggunakan `np.expm1`.

| Model | MAE (Nilai Asli) | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Improved)** 🏆 | **137,137.92** | **$5.66 \times 10^{10}$** | **0.8257 (82.57%)** |
| **Gradient Boosting Regressor** | 138,500.15 | $7.13 \times 10^{10}$ | 0.7805 (78.05%) |

> **Kesimpulan:** Model **Linear Regression** menghasilkan akurasi prediksi tertinggi dengan $R^2$ Score mendekati **82.6%** serta *Mean Absolute Error* (MAE) yang lebih rendah dibandingkan Gradient Boosting. Oleh karena itu, model Linear Regression dipilih dan disimpan sebagai file `car_price_model_improved.joblib`.

---

## Tools dan Library

- **Bahasa Pemrograman:** Python
- **Environment:** Jupyter Notebook
- **Pengolahan & Analisis Data:** Pandas, NumPy
- **Visualisasi Data:** Matplotlib, Seaborn
- **Machine Learning & Modeling:** Scikit-Learn (`LinearRegression`, `GradientBoostingRegressor`, `StandardScaler`, `OrdinalEncoder`)
- **Penyimpanan Model:** Joblib

---

## Cara Menjalankan Proyek

1. **Clone Repository ini:**
   ```bash
   git clone https://github.com/raflyanandyka/vehicle-price-prediction.git
   cd vehicle-price-prediction
   ```

2. **Persiapkan Virtual Environment (Opsional tetapi disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   # venv\Scripts\activate   # Untuk Windows
   ```

3. **Instalasi Library/Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

4. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook vehicle-price-prediction.ipynb
   ```

5. **Eksekusi Cell:** Jalankan seluruh sel di notebook secara berurutan untuk melihat proses pra-pemrosesan, pelatihan model, hingga pembuatan file `car_price_model_improved.joblib`.

---

## Batasan Proyek

- Model ini dilatih menggunakan dataset spesifik dengan tahun acuan maksimal 2020.
- Prediksi sangat dipengaruhi oleh distribusi harga dan merek yang terdaftar di dalam dataset pelatihan.
- Transformasi logaritmik membutuhkan pengembalian fungsi (`np.expm1`) sebelum hasil prediksi dipresentasikan kepada pengguna akhir.

---

## Author

* **Nama:** Muhammad Rafly Anandyka Ekoputra
* **NIM:** 241011402517
* **Kelas:** 05TPLE014
* **Mata Kuliah:** Machine Learning