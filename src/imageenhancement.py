"""
Section 5 — Image Enhancement (Peningkatan Kualitas Citra).

Mendukung:
- Brightness adjustment (pergeseran linier intensitas)
- Contrast adjustment (skalasi kontras terhadap rata-rata atau faktor alpha)
- Gamma Correction (transformasi non-linier power-law s = c * r^gamma)
- Min-Max Intensity Normalization
- Unsharp Masking Enhancement
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def adjust_brightness(image: np.ndarray, beta: int = 50) -> np.ndarray:
    """
    Mengatur kecerahan dengan menambahkan offset konstan:
    I_out = clip(I_in + beta, 0, 255)
    """
    # Gunakan np.clip untuk mencegah integer overflow/underflow
    res = image.astype(np.int32) + beta
    return np.clip(res, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Mengatur kontras dengan penskalaan deviasi dari intensitas tengah (127.5):
    I_out = clip(alpha * (I_in - 127.5) + 127.5, 0, 255)
    """
    res = alpha * (image.astype(np.float32) - 127.5) + 127.5
    return np.clip(res, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """
    Menerapkan Gamma Correction (Power-law transformation):
    gamma < 1.0 mencerahkan bayangan gelap (shadows)
    gamma > 1.0 menggelapkan highlight / meningkatkan kontras area terang.
    Menggunakan Look-Up Table (LUT) untuk efisiensi tinggi.
    """
    if gamma <= 0:
        raise ValueError("Nilai gamma harus positif (> 0).")

    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, lut)


def normalize_min_max(image: np.ndarray) -> np.ndarray:
    """
    Melakukan normalisasi min-max rentang [0, 255].
    """
    img_f = image.astype(np.float32)
    min_v, max_v = img_f.min(), img_f.max()
    if max_v == min_v:
        return image.copy()
    norm = (img_f - min_v) / (max_v - min_v) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def unsharp_mask_enhancement(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5
) -> np.ndarray:
    """
    Meningkatkan detail dan ketajaman lokal melalui unsharp masking:
    Enhanced = Original + strength * (Original - Blurred)
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    sharpened = image.astype(np.float32) + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def visualize_image_enhancement(
    image: np.ndarray,
    brightness_offset: int = 40,
    contrast_factor: float = 1.4,
    gamma_values: Tuple[float, float] = (0.5, 2.0),
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan berbagai teknik image enhancement.
    """
    results: Dict[str, np.ndarray] = {
        "original": image.copy(),
        "brightened": adjust_brightness(image, beta=brightness_offset),
        "darkened": adjust_brightness(image, beta=-brightness_offset),
        "high_contrast": adjust_contrast(image, alpha=contrast_factor),
        "low_contrast": adjust_contrast(image, alpha=0.6),
        "gamma_bright": adjust_gamma(image, gamma=gamma_values[0]),
        "gamma_dark": adjust_gamma(image, gamma=gamma_values[1]),
        "normalized": normalize_min_max(image),
        "unsharp_mask": unsharp_mask_enhancement(image, sigma=1.5, strength=1.5),
    }

    display_images = list(results.values())
    titles = [
        "1. Original",
        f"2. Brightness (+{brightness_offset})",
        f"3. Brightness (-{brightness_offset})",
        f"4. High Contrast (x{contrast_factor})",
        "5. Low Contrast (x0.6)",
        f"6. Gamma Bright (γ={gamma_values[0]})",
        f"7. Gamma Dark (γ={gamma_values[1]})",
        "8. Min-Max Normalization",
        "9. Unsharp Mask Details"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        main_title="Section 5 — Image Enhancement (Kecerahan, Kontras, Gamma, dan Ketajaman)",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
