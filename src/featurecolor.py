"""
Modul Ekstraksi Fitur Warna (Color Features).

Mengimplementasikan analisis distribusi kromatik dan statistik warna:
- Color Histograms (Distribusi 1D per kanal & 2D Hue-Saturation)
- Color Moments (Mean, Standar Deviasi, Skewness pada ruang RGB, HSV, dan LAB)
- Dominant Color Palette (Kuantisasi warna menggunakan K-Means clustering)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_color_moments(channel: np.ndarray) -> Dict[str, float]:
    """
    Menghitung momen warna orde 1 (mean), orde 2 (std), dan orde 3 (skewness).

    Parameters:
        channel: Array 2D kanal warna.

    Returns:
        Dict[str, float]: mean, std, skewness.
    """
    flat = channel.astype(np.float64).ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    # Momen ke-3 (skewness kubik)
    skewness = float(np.mean((flat - mean) ** 3) ** (1.0 / 3.0)) if np.mean((flat - mean) ** 3) >= 0 else float(-((-np.mean((flat - mean) ** 3)) ** (1.0 / 3.0)))

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
    }


def compute_color_features(
    image: np.ndarray,
    n_dominant: int = 5,
    bins_1d: int = 32,
    hist_2d_bins: Tuple[int, int] = (30, 32)
) -> Dict[str, Any]:
    """
    Mengekstrak fitur warna dari citra (Histogram, Momen Warna, dan Palet Dominan).

    Parameters:
        image: Citra input RGB (H, W, 3).
        n_dominant: Jumlah klaster warna dominan (K-Means).
        bins_1d: Jumlah bin untuk histogram 1D.
        hist_2d_bins: Tuple bin (Hue, Saturation) untuk histogram 2D.

    Returns:
        Dict[str, Any]: Momen warna, histogram 1D/2D, dan warna dominan.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Input citra harus berupa numpy.ndarray 3D (RGB).")

    img_rgb = image.copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Konversi ruang warna
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # 1. Color Moments untuk RGB, HSV, dan LAB
    moments = {
        "rgb": {
            "R": compute_color_moments(img_rgb[:, :, 0]),
            "G": compute_color_moments(img_rgb[:, :, 1]),
            "B": compute_color_moments(img_rgb[:, :, 2]),
        },
        "hsv": {
            "H": compute_color_moments(img_hsv[:, :, 0]),
            "S": compute_color_moments(img_hsv[:, :, 1]),
            "V": compute_color_moments(img_hsv[:, :, 2]),
        },
        "lab": {
            "L": compute_color_moments(img_lab[:, :, 0]),
            "A": compute_color_moments(img_lab[:, :, 1]),
            "B_channel": compute_color_moments(img_lab[:, :, 2]),
        }
    }

    # 2. Histogram 1D (RGB dan HSV)
    hists_1d = {}
    for i, c_name in enumerate(["R", "G", "B"]):
        hist = cv2.calcHist([img_rgb], [i], None, [bins_1d], [0, 256])
        hists_1d[c_name] = (hist / (hist.sum() + 1e-8)).flatten()

    for i, (c_name, r_max) in enumerate([("H", 180), ("S", 256), ("V", 256)]):
        hist = cv2.calcHist([img_hsv], [i], None, [bins_1d], [0, r_max])
        hists_1d[c_name] = (hist / (hist.sum() + 1e-8)).flatten()

    # 3. 2D Joint Histogram Hue-Saturation
    hist_2d_hs = cv2.calcHist(
        [img_hsv],
        [0, 1],
        None,
        list(hist_2d_bins),
        [0, 180, 0, 256]
    )
    hist_2d_hs = cv2.normalize(hist_2d_hs, None, 0, 1, cv2.NORM_MINMAX)

    # 4. K-Means Dominant Colors
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        n_dominant,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    labels = labels.flatten()
    
    counts = np.bincount(labels, minlength=n_dominant)
    percentages = counts / len(labels)
    # Urutkan berdasarkan persentase tertinggi
    sort_idx = np.argsort(-percentages)
    dominant_palette = centers[sort_idx]
    dominant_percentages = percentages[sort_idx]

    # Rangkum semua momen warna ke dalam satu vektor numerik 1D (9 nilai: mean, std, skewness x 3 channel)
    color_vector = []
    for sp in ["rgb", "hsv", "lab"]:
        for ch in moments[sp].values():
            color_vector.extend([ch["mean"], ch["std"], ch["skewness"]])

    return {
        "rgb": img_rgb,
        "hsv": img_hsv,
        "lab": img_lab,
        "moments": moments,
        "color_vector": np.array(color_vector, dtype=np.float32),
        "hists_1d": hists_1d,
        "hist_2d_hs": hist_2d_hs,
        "dominant_palette": dominant_palette,
        "dominant_percentages": dominant_percentages,
    }


def visualize_color_features(
    image: np.ndarray,
    n_dominant: int = 5,
    figsize: Tuple[int, int] = (15, 9),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Memvisualisasikan histogram warna 1D, 2D H-S heatmap, dan palet warna dominan.

    Parameters:
        image: Citra input RGB.
        n_dominant: Jumlah warna dominan.
        figsize: Dimensi figure matplotlib.
        show: Jika True, langsung menampilkan figure dengan plt.show().
        save_path: Path penyimpanan figure (opsional).

    Returns:
        Dict[str, Any]: Hasil ekstraksi fitur warna dan figure matplotlib.
    """
    res = compute_color_features(image, n_dominant=n_dominant)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Citra Asli & Palet Dominan
    ax0 = axes[0, 0]
    ax0.imshow(res["rgb"])
    ax0.set_title("1. Citra Input RGB", fontsize=11, fontweight="bold")
    ax0.axis("off")

    # 2. Histogram 1D Kanal RGB
    ax1 = axes[0, 1]
    colors = {"R": "crimson", "G": "forestgreen", "B": "royalblue"}
    bins = np.linspace(0, 255, len(res["hists_1d"]["R"]))
    for c_name, col in colors.items():
        ax1.plot(bins, res["hists_1d"][c_name], color=col, label=f"Kanal {c_name}", linewidth=2)
    ax1.set_title("2. Histogram 1D (RGB Distribution)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Nilai Intensitas (0-255)", fontsize=9)
    ax1.set_ylabel("Frekuensi Normalisasi", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 3. 2D Joint Histogram (Hue-Saturation)
    ax2 = axes[1, 0]
    im_hs = ax2.imshow(res["hist_2d_hs"], interpolation="nearest", origin="lower", cmap="viridis")
    ax2.set_title("3. 2D Joint Histogram (Hue-Saturation)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Saturation Bins", fontsize=9)
    ax2.set_ylabel("Hue Bins", fontsize=9)
    plt.colorbar(im_hs, ax=ax2, fraction=0.046, pad=0.04)

    # 4. Palet Warna Dominan (K-Means)
    ax3 = axes[1, 1]
    # Buat bar horizontal berurutan sesuai proporsi warna
    left = 0
    for col, pct in zip(res["dominant_palette"], res["dominant_percentages"]):
        col_norm = col / 255.0
        ax3.barh(0, pct, left=left, color=col_norm, edgecolor="black", height=0.5)
        # Berikan teks persentase jika porsi > 5%
        if pct > 0.05:
            ax3.text(
                left + pct / 2, 0,
                f"{pct*100:.1f}%\nRGB{tuple(col)}",
                ha="center", va="center",
                color="white" if np.mean(col) < 128 else "black",
                fontsize=8, fontweight="bold"
            )
        left += pct

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([])
    ax3.set_xlabel("Proporsi Kumulatif", fontsize=9)
    ax3.set_title(f"4. Palet {n_dominant} Warna Dominan (K-Means Clustering)", fontsize=11, fontweight="bold")

    plt.suptitle("Feature Extraction: Color Descriptors, Moments, & Distributions", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    res["fig"] = fig
    return res
