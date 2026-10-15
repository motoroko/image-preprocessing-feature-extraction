"""
Section 10 — Resolusi, Skala, dan Piramida Citra (Image Pyramids & Multiresolution).

Mendukung:
- Downsampling & Upsampling
- Perbandingan Metode Interpolasi (Nearest, Bilinear, Bicubic, Lanczos)
- Gaussian Pyramid (Dekomposisi multiresolusi low-pass)
- Laplacian Pyramid (Dekomposisi detail frekuensi tinggi band-pass)
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def compare_interpolations(
    image: np.ndarray,
    scale_factor: float = 0.25
) -> Dict[str, np.ndarray]:
    """
    Membandingkan efek berbagai algoritma interpolasi saat downscale dan upscale kembali.
    """
    h, w = image.shape[:2]
    small_size = (max(1, int(w * scale_factor)), max(1, int(h * scale_factor)))

    methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }

    results = {}
    for name, flag in methods.items():
        down = cv2.resize(image, small_size, interpolation=flag)
        up = cv2.resize(down, (w, h), interpolation=flag)
        results[name] = up

    return results


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """Membangun piramida Gaussian: berulang kali melakukan blur dan downsampling 2x."""
    pyramid = [image.copy()]
    current = image.copy()
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Membangun piramida Laplacian: merekam perbedaan/residual detail frekuensi tinggi.
    L_i = G_i - pyrUp(G_{i+1})
    """
    gauss_pyr = build_gaussian_pyramid(image, levels=levels)
    laplacian_pyr = []

    for i in range(levels - 1):
        size = (gauss_pyr[i].shape[1], gauss_pyr[i].shape[0])
        expanded = cv2.pyrUp(gauss_pyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gauss_pyr[i], expanded)
        laplacian_pyr.append(laplacian)

    # Tingkat teratas menyimpan Gaussian level terkecil
    laplacian_pyr.append(gauss_pyr[-1])
    return laplacian_pyr


def visualize_image_pyramids(
    image: np.ndarray,
    levels: int = 4,
    scale_factor: float = 0.2,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan interpolasi dan visualisasi piramida multiresolusi.
    """
    interp_res = compare_interpolations(image, scale_factor=scale_factor)
    gauss_pyr = build_gaussian_pyramid(image, levels=levels)
    lap_pyr = build_laplacian_pyramid(image, levels=levels)

    display_images: List[np.ndarray] = []
    titles: List[str] = []

    # 1. Interpolasi comparison
    for name, up_img in interp_res.items():
        display_images.append(up_img)
        titles.append(f"Interpolation: {name.capitalize()}")

    # 2. Gaussian Pyramid levels (di-resize ke ukuran asli untuk tampilan grid seragam)
    h, w = image.shape[:2]
    for idx, g_img in enumerate(gauss_pyr):
        gh, gw = g_img.shape[:2]
        # Pad / resize agar visualisasi grid konsisten
        vis_g = cv2.resize(g_img, (w, h), interpolation=cv2.INTER_NEAREST)
        display_images.append(vis_g)
        titles.append(f"Gaussian L{idx} ({gw}x{gh})")

    # 3. Laplacian Pyramid levels
    for idx, l_img in enumerate(lap_pyr):
        lh, lw = l_img.shape[:2]
        vis_l = cv2.resize(l_img, (w, h), interpolation=cv2.INTER_NEAREST)
        # Normalisasi kontras agar residual detail terlihat jelas
        vis_l_norm = cv2.normalize(vis_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        display_images.append(vis_l_norm)
        titles.append(f"Laplacian L{idx} ({lw}x{lh})")

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize if figsize else (15, 9),
        main_title="Section 10 — Multiresolusi: Interpolasi, Gaussian & Laplacian Pyramid",
        show=show
    )

    return {
        "interpolations": interp_res,
        "gaussian_pyramid": gauss_pyr,
        "laplacian_pyramid": lap_pyr,
        "figure": fig
    }
