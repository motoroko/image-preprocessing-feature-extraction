"""
Section 4 — Histogram dan Distribusi Intensitas (Histogram Processing).

Mendukung:
- Perhitungan histogram RGB dan Grayscale
- Histogram Equalization (Global)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) pada kanal Luminans citra berwarna
- Contrast Stretching (Linear Min-Max percentile stretching)
- Perbandingan komprehensif citra sebelum dan sesudah pemrosesan beserta kurva histogramnya.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_comparison_with_histogram


def compute_histogram(
    image: np.ndarray,
    bins: int = 256
) -> Dict[str, np.ndarray]:
    """
    Menghitung data histogram intensitas.
    Untuk grayscale: dict {'gray': hist}
    Untuk RGB: dict {'r': hist_r, 'g': hist_g, 'b': hist_b}
    """
    if image.ndim == 2:
        hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, 256))
        return {"gray": hist}
    else:
        return {
            "r": np.histogram(image[:, :, 0].ravel(), bins=bins, range=(0, 256))[0],
            "g": np.histogram(image[:, :, 1].ravel(), bins=bins, range=(0, 256))[0],
            "b": np.histogram(image[:, :, 2].ravel(), bins=bins, range=(0, 256))[0]
        }


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan Global Histogram Equalization.
    Untuk citra berwarna RGB, equalization diterapkan HANYA pada kanal Luminans (Y dari YCrCb)
    agar tidak merusak keseimbangan hue dan saturation warna asli.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    else:
        # Konversi ke YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Untuk citra berwarna, diterapkan pada kanal L pada ruang warna LAB atau Y pada YCrCb
    untuk meningkatkan kontras lokal tanpa menimbulkan artefak warna berlebih.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_contrast_stretching(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0
) -> np.ndarray:
    """
    Melakukan normalisasi peregangan kontras linier berdasarkan persentil intensitas.
    """
    if image.ndim == 2:
        p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
        if p_high == p_low:
            return image.copy()
        stretched = (image.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)
    else:
        res = np.zeros_like(image)
        for c in range(3):
            ch = image[:, :, c]
            p_low, p_high = np.percentile(ch, (low_percentile, high_percentile))
            if p_high == p_low:
                res[:, :, c] = ch
            else:
                stretched = (ch.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
                res[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
        return res


def visualize_histogram_processing(
    image: np.ndarray,
    clahe_clip_limit: float = 2.5,
    figsize: Tuple[int, int] = (14, 10),
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan Histogram Equalization, CLAHE, dan Contrast Stretching
    beserta histogram distribusi intensitas masing-masing.
    """
    hist_eq = apply_histogram_equalization(image)
    clahe_img = apply_clahe(image, clip_limit=clahe_clip_limit)
    stretch_img = apply_contrast_stretching(image)

    methods = [
        ("Original", image),
        ("Global Hist Equalization", hist_eq),
        (f"CLAHE (clip={clahe_clip_limit})", clahe_img),
        ("Contrast Stretching (2-98%)", stretch_img)
    ]

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    for col_idx, (name, img_data) in enumerate(methods):
        # Baris 0: Citra
        ax_img = axes[0, col_idx]
        if img_data.ndim == 2:
            ax_img.imshow(img_data, cmap="gray")
        else:
            ax_img.imshow(img_data)
        ax_img.set_title(name, fontsize=11, weight="bold")
        ax_img.axis("off")

        # Baris 1: Histogram
        ax_hist = axes[1, col_idx]
        if img_data.ndim == 2:
            ax_hist.hist(img_data.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
        else:
            colors = ("red", "green", "blue")
            for c_idx, color in enumerate(colors):
                ax_hist.hist(img_data[:, :, c_idx].ravel(), bins=256, range=(0, 256), color=color, alpha=0.45, label=color[0].upper())
            if col_idx == 0:
                ax_hist.legend(loc="upper right", fontsize=8)
        ax_hist.set_title(f"Histogram of {name}", fontsize=9)
        ax_hist.set_xlim([0, 256])
        ax_hist.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Section 4 — Histogram Processing dan Distribusi Intensitas", fontsize=14, weight="bold", y=0.98)
    fig.tight_layout()

    if show:
        plt.show()

    return {
        "results": {
            "original": image.copy(),
            "equalization": hist_eq,
            "clahe": clahe_img,
            "stretching": stretch_img
        },
        "figure": fig
    }
