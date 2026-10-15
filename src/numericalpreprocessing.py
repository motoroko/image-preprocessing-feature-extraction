"""
Section 11 — Normalisasi dan Representasi Numerik (Numerical Preprocessing).

Mendukung transformasi representasi numerik data citra:
- Konversi uint8 [0, 255] ke float32
- Normalisasi Min-Max ke rentang [0.0, 1.0]
- Normalisasi rentang simetris [-1.0, 1.0]
- Standardisasi Z-Score (Zero mean, unit variance: (x - mu) / sigma)
- Penanganan citra konstan / deviasi nol (zero division safety)
- Inspeksi mendalam statistik numerik (shape, dtype, min, max, mean, std)
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def convert_uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Mengonversi array citra uint8 ke float32."""
    return image.astype(np.float32)


def normalize_to_0_1(image: np.ndarray) -> np.ndarray:
    """
    Melakukan normalisasi rentang [0.0, 1.0].
    I_norm = (I - min) / (max - min)
    Aman terhadap citra konstan (pembagian dengan nol dicegah).
    """
    img_f = image.astype(np.float32)
    min_v, max_v = float(np.min(img_f)), float(np.max(img_f))
    diff = max_v - min_v
    if diff == 0:
        return np.zeros_like(img_f)
    return (img_f - min_v) / diff


def normalize_to_minus1_1(image: np.ndarray) -> np.ndarray:
    """
    Melakukan normalisasi rentang simetris [-1.0, 1.0].
    I_norm = 2 * ((I - min) / (max - min)) - 1.0
    Aman terhadap citra konstan.
    """
    zero_one = normalize_to_0_1(image)
    return 2.0 * zero_one - 1.0


def standardize_zscore(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Melakukan standardisasi Z-score: (I - mu) / sigma.
    Aman terhadap citra konstan (jika sigma == 0, mengembalikan array 0).
    """
    img_f = image.astype(np.float32)
    mean_val = float(np.mean(img_f))
    std_val = float(np.std(img_f))
    if std_val == 0:
        return np.zeros_like(img_f), mean_val, std_val
    return (img_f - mean_val) / std_val, mean_val, std_val


def get_array_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Mengambil ringkasan statistik numerik array."""
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4)
    }


def visualize_numerical_preprocessing(
    image: np.ndarray,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan transformasi representasi numerik.
    Menampilkan citra display yang dipetakan secara aman beserta tabel statistik numeriknya.
    """
    img_u8 = image.copy()
    img_0_1 = normalize_to_0_1(img_u8)
    img_neg1_1 = normalize_to_minus1_1(img_u8)
    img_zscore, z_mean, z_std = standardize_zscore(img_u8)

    stats = {
        "Original (uint8)": get_array_stats(img_u8),
        "Normalized [0, 1]": get_array_stats(img_0_1),
        "Normalized [-1, 1]": get_array_stats(img_neg1_1),
        "Standardized Z-Score": get_array_stats(img_zscore)
    }

    # Untuk kebutuhan display visual matplotlib yang aman
    # Matplotlib imshow menerima float32 hanya pada rentang [0, 1]
    disp_neg1 = (img_neg1_1 + 1.0) / 2.0
    disp_z = (img_zscore - img_zscore.min()) / (img_zscore.max() - img_zscore.min() + 1e-7)

    display_images = [img_u8, img_0_1, disp_neg1, disp_z]
    titles = [
        f"1. Original uint8\n[{stats['Original (uint8)']['min']}, {stats['Original (uint8)']['max']}]",
        f"2. Normalized [0, 1]\n[{stats['Normalized [0, 1]']['min']}, {stats['Normalized [0, 1]']['max']}]",
        f"3. Normalized [-1, 1] (Mapped)\n[{stats['Normalized [-1, 1]']['min']}, {stats['Normalized [-1, 1]']['max']}]",
        f"4. Z-score (Mapped for display)\nμ={stats['Standardized Z-Score']['mean']}, σ={stats['Standardized Z-Score']['std']}"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=4,
        figsize=figsize if figsize else (15, 4.5),
        main_title="Section 11 — Transformasi Representasi Numerik & Statistik Citra",
        show=show
    )

    return {
        "stats": stats,
        "arrays": {
            "uint8": img_u8,
            "norm_0_1": img_0_1,
            "norm_neg1_1": img_neg1_1,
            "zscore": img_zscore
        },
        "figure": fig
    }
