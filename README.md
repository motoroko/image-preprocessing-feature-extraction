# Image Preprocessing Playground

Repositori akademik untuk eksplorasi, komparasi, dan visualisasi berbagai teknik mendasar *image preprocessing* (pengolahan citra digital) menggunakan Python dan Jupyter Notebook untuk mata kuliah **Pengenalan Pola** (*Pattern Recognition*).

Fokus utama proyek ini adalah **pemahaman konseptual, karakteristik matematis transformasi, dan inspeksi visual**. Proyek ini **tidak** membangun atau melatih model klasifikasi, neural network, atau memprediksi emosi.

---

## Daftar Isi
- [Struktur Direktori](#struktur-direktori)
- [Prasyarat Sistem](#prasyarat-sistem)
- [Instalasi dan Setup Lingkungan](#instalasi-dan-setup-lingkungan)
- [Pengunduhan Dataset Kaggle & Konfigurasi Path](#pengunduhan-dataset-kaggle--konfigurasi-path)
- [Cara Menjalankan Notebook](#cara-menjalankan-notebook)
- [Menjalankan Unit Test](#menjalankan-unit-test)
- [Ringkasan 12 Section Preprocessing](#ringkasan-12-section-preprocessing)
- [Batasan Proyek](#batasan-proyek)

---

## Struktur Direktori

```text
image-preprocessing-playground/
├── main.ipynb                    # Notebook utama eksperimen & visualisasi
├── requirements.txt              # Daftar dependensi Python
├── README.md                     # Dokumentasi panduan proyek
├── .gitignore                    # Konfigurasi ignore file cache dan virtualenv
├── data/                         # Folder dataset gambar
│   └── .gitkeep
├── src/                          # Modul algoritma Python modular
│   ├── __init__.py
│   ├── config.py                 # Konfigurasi konstanta & path terpusat
│   ├── dataloader.py             # Pemuatan gambar, validasi format, generator demo
│   ├── visualization.py          # Helper visualisasi grid subplot & histogram
│   ├── colorspaces.py            # Section 1: Ruang warna (RGB, HSV, LAB, dll.)
│   ├── geometrictransformations.py # Section 2: Transformasi geometris
│   ├── imagefiltering.py         # Section 3: Filtering spasial & deteksi tepi
│   ├── histogramprocessing.py    # Section 4: Histogram equalization, CLAHE
│   ├── imageenhancement.py       # Section 5: Kecerahan, kontras, gamma, unsharp
│   ├── morphologicaloperations.py# Section 6: Morfologi erosi, dilasi, tophat
│   ├── noiseprocessing.py        # Section 7: Noise & denoising (MSE/PSNR)
│   ├── thresholding.py           # Section 8: Thresholding Otsu & adaptive
│   ├── segmentation.py           # Section 9: K-Means warna, kontur, watershed
│   ├── imagepyramids.py          # Section 10: Interpolasi & multiresolusi
│   ├── numericalpreprocessing.py # Section 11: Normalisasi [0,1], [-1,1], Z-score
│   ├── augmentation.py           # Section 12: Augmentasi data terkontrol
│   ├── featureedge.py            # Section 13: Edge detection & gradient orientation
│   ├── featurekeypoint.py        # Section 14: Corner & keypoint descriptors
│   ├── featuretexture.py         # Section 15: Analisis tekstur GLCM, LBP, Gabor
│   ├── featurecolor.py           # Section 16: Momen warna, histogram 1D/2D, K-Means
│   └── featureshape.py           # Section 17: Deskriptor bentuk, Hu moments, HOG
└── tests/                        # Pengujian unit test (pytest)
    ├── conftest.py               # Konfigurasi backend headless matplotlib
    ├── testdataloader.py
    ├── testcolorspaces.py
    ├── testgeometrictransformations.py
    ├── test_preprocessing.py
    └── test_feature_extraction.py
```

---

## Prasyarat Sistem
- Python 3.10 atau 3.11+
- Git

---

## Instalasi dan Setup Lingkungan

1. **Clone repositori**:

2. **Buat dan aktifkan Virtual Environment**:
   - Di Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - Di Linux / macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Pasang Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pengunduhan Dataset Kaggle & Konfigurasi Path

Dataset yang diacu adalah:  
**[Facial Emotion Recognition Images Raw Dataset](https://www.kaggle.com/datasets/frmgtx/facial-emotion-recognition-images-raw-dataset)**

### Cara Pengunduhan:
1. **Otomatis via Notebook / Script (Direkomendasikan)**:
   - Pada bagian 0.1 di `main.ipynb`, jika direktori `data/` masih kosong dan variabel `user_synthetic_data = False`, sistem akan otomatis mengunduh dataset dari Kaggle via `kagglehub` dan mengekstraknya ke folder `data/`.
2. **Menggunakan Kaggle CLI** (jika telah memiliki `kaggle.json`):
   ```bash
   kaggle datasets download -d frmgtx/facial-emotion-recognition-images-raw-dataset -p data/ --unzip
   ```
3. **Pengunduhan Manual**:
   - Buka tautan dataset di atas melalui browser.
   - Klik tombol **Download** (format `.zip`).
   - Ekstrak seluruh isi file gambar ke dalam folder `data/` pada proyek ini.

> **Skema Pemilihan Data Citra (`user_synthetic_data`):**  
> Pada `main.ipynb` Section 0.1, tersedia variabel kendali `user_synthetic_data`:
> - `user_synthetic_data = False`: Menggunakan citra nyata dari dataset Kaggle (mengunduh & mengekstrak otomatis jika direktori kosong).
> - `user_synthetic_data = True`: Menggunakan citra wajah sintetis terstruktur (`create_synthetic_face()`) beresolusi 256×256 secara instan tanpa perlu unduhan.

---

## Cara Menjalankan Notebook

Jalankan server Jupyter Notebook / JupyterLab:
```bash
jupyter notebook main.ipynb
```
Atau buka file `main.ipynb` langsung di editor seperti VS Code atau IDE yang mendukung ekstensi notebook Jupyter.

Seluruh sel di dalam [main.ipynb](file:///d:/1-S2/Semester%203/Pattern%20Recognition/Tugas%20Pengolahan%20Citra/main.ipynb) dirancang untuk dieksekusi secara berurutan (*Run All*) dari sel pertama hingga selesai.

---

## Menjalankan Unit Test

Pengujian modular dilakukan menggunakan `pytest`. Pengujian menggunakan backend non-interaktif (`Agg`) dan citra uji sintetis kecil sehingga tidak bergantung pada koneksi internet ataupun file dataset eksternal.

Jalankan perintah berikut:
```bash
python -m pytest tests/
```
Output yang diharapkan:
```text
============================= 32 passed in 5.75s ==============================
```

---

## Ringkasan Section Eksperimen

### Bagian 1: Image Preprocessing (Section 1 - 12)
1. **Representasi & Ruang Warna (`src/colorspaces.py`):** Konversi RGB, BGR, Grayscale, HSV, HLS, LAB, XYZ, YCrCb, YUV, dan CMYK.
2. **Transformasi Geometris (`src/geometrictransformations.py`):** Crop, resize, rotasi, translasi, flip, zoom, shear, affine, dan perspektif homografi.
3. **Filtering & Konvolusi (`src/imagefiltering.py`):** Box, Gaussian, Median, Bilateral, Sharpening, Sobel, Laplacian, dan custom 2D kernel.
4. **Histogram Processing (`src/histogramprocessing.py`):** Histogram kanal warna, Global Histogram Equalization, CLAHE, dan Contrast Stretching.
5. **Image Enhancement (`src/imageenhancement.py`):** Brightness ($\beta$), Contrast ($\alpha$), Gamma correction, dan Unsharp masking.
6. **Operasi Morfologi (`src/morphologicaloperations.py`):** Erosi, Dilasi, Opening, Closing, Morphological Gradient, Top-Hat, Black-Hat.
7. **Noise & Denoising (`src/noiseprocessing.py`):** Gaussian, Salt-and-Pepper, Speckle noise; evaluasi kuantitatif MSE dan PSNR.
8. **Thresholding & Binarisasi (`src/thresholding.py`):** Global binary, Otsu thresholding otomatis, dan Adaptive thresholding (Mean & Gaussian).
9. **Segmentasi Citra (`src/segmentation.py`):** Masking threshold, K-Means color clustering, kontur biner, dan Watershed segmentation.
10. **Multiresolusi & Piramida Citra (`src/imagepyramids.py`):** Interpolasi resolusi, Gaussian Pyramid, dan Laplacian Pyramid.
11. **Representasi Numerik (`src/numericalpreprocessing.py`):** Normalisasi $[0.0, 1.0]$, penskalaan $[-1.0, 1.0]$, dan standardisasi Z-Score.
12. **Augmentasi Citra (`src/augmentation.py`):** Transformasi stokastik acak dan pipeline kombinasi dengan random seed terkontrol.

### Bagian 2: Feature Extraction (Section 13 - 18)
13. **Edge Detection (`src/featureedge.py`):** Sobel, Prewitt, Scharr, Laplacian of Gaussian (LoG), Canny edge detector, magnitudo, dan arah gradien.
14. **Corner & Keypoint Detection (`src/featurekeypoint.py`):** Harris Corner, Shi-Tomasi, FAST, SIFT (128D DoG keypoint), dan ORB binary descriptors.
15. **Tekstur / Texture Features (`src/featuretexture.py`):** Matriks ko-okurensi GLCM (properti Haralick), LBP (Local Binary Patterns) uniform, dan Gabor Filter Bank multi-orientasi.
16. **Warna / Color Descriptors (`src/featurecolor.py`):** Histogram 1D (RGB/HSV), 2D Joint Histogram ($H\text{-}S$), Color Moments (Mean, Std, Skewness pada RGB, HSV, LAB), dan K-Means Dominant Colors.
17. **Shape & Geometry (`src/featureshape.py`):** Deskriptor kontur (Area, Perimeter, Circularity, Solidity, Bounding Box), 7 Hu Invariant Moments, Hough Transform (Lines & Circles), dan HOG (Histogram of Oriented Gradients).
18. **Sintesis Vektor Fitur Terpadu:** Agregasi seluruh modalitas fitur ke dalam satu vektor numerik 1D ($D = 2.162$ dimensi) siap klasifikasi.

---

## Batasan Proyek
- Proyek ini berorientasi pada pemahaman konsep dasar pengolahan citra digital.
- Tidak mencakup pelatihan arsitektur Deep Learning (CNN/Vision Transformer) atau pembuatan classifier emosi.
- Parameter setiap fungsi dapat dimodifikasi secara dinamis langsung melalui argumen pemanggilan fungsi pada notebook `main.ipynb`.
