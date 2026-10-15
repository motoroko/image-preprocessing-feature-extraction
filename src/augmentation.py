"""
Section 12 — Augmentasi Citra dan Eksperimen Gabungan (Data Augmentation).

Mendukung pipeline augmentasi stokastik/terkontrol:
- Horizontal Flip (Mirroring)
- Random Small Rotation
- Brightness & Contrast Jittering
- Random Scaling / Zoom
- Random Translation (Shift)
- Combined Multi-Transform Pipeline (Komposit realistis)
Reproducibility dijamin penuh menggunakan seed generator acak.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def augment_horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Membalik citra secara horizontal."""
    return cv2.flip(image, 1)


def augment_small_rotation(
    image: np.ndarray,
    max_angle: float = 15.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Memutar citra dengan sudut kecil acak [-max_angle, max_angle]."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def augment_brightness_contrast(
    image: np.ndarray,
    alpha_range: Tuple[float, float] = (0.8, 1.25),
    beta_range: Tuple[int, int] = (-25, 25),
    seed: Optional[int] = None
) -> np.ndarray:
    """Mengubah kontras dan kecerahan secara acak dalam batas aman."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(alpha_range[0], alpha_range[1])
    beta = rng.integers(beta_range[0], beta_range[1])
    res = alpha * image.astype(np.float32) + beta
    return np.clip(res, 0, 255).astype(np.uint8)


def augment_zoom(
    image: np.ndarray,
    zoom_range: Tuple[float, float] = (0.9, 1.15),
    seed: Optional[int] = None
) -> np.ndarray:
    """Melakukan scaling/zoom acak dengan mempertahankan ukuran canvas."""
    rng = np.random.default_rng(seed)
    factor = rng.uniform(zoom_range[0], zoom_range[1])
    h, w = image.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if factor >= 1.0:
        sy = (new_h - h) // 2
        sx = (new_w - w) // 2
        return resized[sy:sy + h, sx:sx + w].copy()
    else:
        pt = (h - new_h) // 2
        pb = h - new_h - pt
        pl = (w - new_w) // 2
        pr = w - new_w - pl
        return cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_REFLECT)


def augment_translation(
    image: np.ndarray,
    max_shift: float = 0.08,
    seed: Optional[int] = None
) -> np.ndarray:
    """Menggeser citra secara acak sejauh fraksi dimensi."""
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    dx = rng.uniform(-max_shift, max_shift) * w
    dy = rng.uniform(-max_shift, max_shift) * h
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def augment_pipeline_combined(
    image: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Menerapkan pipeline kombinasi realistis dari beberapa augmentasi.
    """
    res = image.copy()
    rng = np.random.default_rng(seed)

    if rng.random() > 0.5:
        res = augment_horizontal_flip(res)

    res = augment_small_rotation(res, max_angle=12.0, seed=rng.integers(0, 10000))
    res = augment_brightness_contrast(res, seed=rng.integers(0, 10000))
    res = augment_translation(res, max_shift=0.06, seed=rng.integers(0, 10000))
    return res


def visualize_augmentations(
    image: np.ndarray,
    n_samples: int = 3,
    seed: int = 42,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk memvisualisasikan augmentasi individu dan variasi kombinasi stokastik.
    """
    results: Dict[str, np.ndarray] = {
        "original": image.copy(),
        "flip_h": augment_horizontal_flip(image),
        "small_rotation": augment_small_rotation(image, max_angle=15.0, seed=seed),
        "brightness_contrast": augment_brightness_contrast(image, seed=seed),
        "zoom": augment_zoom(image, seed=seed),
        "translation": augment_translation(image, seed=seed)
    }

    # Generate n_samples kombinasi
    for i in range(1, n_samples + 1):
        results[f"combined_sample_{i}"] = augment_pipeline_combined(image, seed=seed + i * 10)

    display_images = list(results.values())
    titles = [
        "1. Original Baseline",
        "2. Horizontal Flip",
        "3. Small Rotation (±15°)",
        "4. Brightness & Contrast",
        "5. Random Zoom",
        "6. Random Translation",
        "7. Combined Pipeline #1",
        "8. Combined Pipeline #2",
        "9. Combined Pipeline #3"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        main_title="Section 12 — Augmentasi Citra dan Eksperimen Stokastik Gabungan",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
