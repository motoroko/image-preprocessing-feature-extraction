"""
Section 8 — Thresholding dan Binarisasi (Thresholding & Binarization).

Mendukung:
- Global Fixed Thresholding:
  * Binary Thresholding
  * Binary Inverse
  * Truncate
  * To-Zero
  * To-Zero Inverse
- Automatic Global Thresholding:
  * Otsu's Thresholding (memaksimalkan varians antarkelas)
- Local Adaptive Thresholding:
  * Adaptive Mean Thresholding
  * Adaptive Gaussian Thresholding
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def apply_global_threshold(
    image_gray: np.ndarray,
    thresh_value: int = 127,
    max_value: int = 255,
    thresh_type: str = "binary"
) -> Tuple[float, np.ndarray]:
    """Menerapkan thresholding global fixed."""
    type_map = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV
    }
    flag = type_map.get(thresh_type.lower(), cv2.THRESH_BINARY)
    ret, res = cv2.threshold(image_gray, thresh_value, max_value, flag)
    return ret, res


def apply_otsu_threshold(
    image_gray: np.ndarray,
    max_value: int = 255
) -> Tuple[float, np.ndarray]:
    """Menerapkan algoritma Otsu untuk mencari nilai ambang optimal secara otomatis."""
    ret, res = cv2.threshold(image_gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, res


def apply_adaptive_threshold(
    image_gray: np.ndarray,
    method: str = "gaussian",
    block_size: int = 11,
    C: int = 2,
    max_value: int = 255
) -> np.ndarray:
    """Menerapkan adaptive thresholding lokal (Mean atau Gaussian)."""
    # block_size harus bilangan ganjil > 1
    bs = block_size if block_size % 2 == 1 else block_size + 1
    bs = max(3, bs)

    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(image_gray, max_value, adaptive_method, cv2.THRESH_BINARY, bs, C)


def visualize_thresholding(
    image: np.ndarray,
    threshold: int = 127,
    block_size: int = 11,
    C: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan thresholding global, Otsu, dan adaptive.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    _, t_bin = apply_global_threshold(gray, thresh_value=threshold, thresh_type="binary")
    _, t_bin_inv = apply_global_threshold(gray, thresh_value=threshold, thresh_type="binary_inv")
    _, t_trunc = apply_global_threshold(gray, thresh_value=threshold, thresh_type="trunc")
    _, t_tozero = apply_global_threshold(gray, thresh_value=threshold, thresh_type="tozero")
    _, t_tozero_inv = apply_global_threshold(gray, thresh_value=threshold, thresh_type="tozero_inv")

    otsu_val, t_otsu = apply_otsu_threshold(gray)
    t_adapt_mean = apply_adaptive_threshold(gray, method="mean", block_size=block_size, C=C)
    t_adapt_gauss = apply_adaptive_threshold(gray, method="gaussian", block_size=block_size, C=C)

    results: Dict[str, np.ndarray] = {
        "grayscale": gray.copy(),
        "binary": t_bin,
        "binary_inv": t_bin_inv,
        "trunc": t_trunc,
        "tozero": t_tozero,
        "tozero_inv": t_tozero_inv,
        "otsu": t_otsu,
        "adaptive_mean": t_adapt_mean,
        "adaptive_gaussian": t_adapt_gauss
    }

    display_images = list(results.values())
    titles = [
        "1. Grayscale Input",
        f"2. Binary (T={threshold})",
        f"3. Binary Inv (T={threshold})",
        f"4. Truncate (T={threshold})",
        f"5. To-Zero (T={threshold})",
        f"6. To-Zero Inv (T={threshold})",
        f"7. Otsu (Optimal T={int(otsu_val)})",
        f"8. Adaptive Mean (block={block_size})",
        f"9. Adaptive Gaussian (block={block_size})"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        cmaps="gray",
        main_title="Section 8 — Thresholding dan Binarisasi Citra",
        show=show
    )

    return {
        "results": results,
        "otsu_threshold_value": otsu_val,
        "figure": fig
    }
