"""
Modul Deteksi Tepi (Edge Detection) untuk Ekstraksi Fitur Citra.

Mengimplementasikan metode deteksi tepi dan perhitungan gradien spasial klasik:
- Sobel Operator (Gradien X, Gradien Y, Magnitudo, Sudut Arah)
- Prewitt Operator (Konvolusi kernel diferensiasi ortogonal)
- Scharr Operator (Turunan pertama presisi tinggi)
- Laplacian of Gaussian / LoG (Deteksi zero-crossing turunan kedua)
- Canny Edge Detector (Deteksi tepi optimal multi-tahap dengan histeresis)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Helper internal untuk memastikan input berupa 2D grayscale uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input citra harus berupa numpy.ndarray.")
    
    if image.ndim == 3:
        # Asumsikan input RGB sesuai konvensi proyek
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Dimensi array citra tidak didukung (harus 2D atau 3D).")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def compute_edge_maps(
    image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    blur_ksize: int = 3,
    sobel_ksize: int = 3
) -> Dict[str, Any]:
    """
    Menghitung peta tepi dan turunan spasial menggunakan 5 algoritma berbeda.

    Parameters:
        image: Array citra (2D Gray atau 3D RGB).
        low_thresh: Ambang bawah histeresis Canny.
        high_thresh: Ambang atas histeresis Canny.
        blur_ksize: Ukuran kernel Gaussian blur untuk reduksi noise sebelum diferensiasi.
        sobel_ksize: Ukuran kernel Sobel (1, 3, 5, 7).

    Returns:
        Dict[str, Any]: Peta tepi uint8, peta magnitudo, sudut gradien, dan kepadatan tepi.
    """
    gray = _to_gray_uint8(image)
    
    # Preprocessing: Gaussian smoothing untuk menekan noise frekuensi tinggi
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        blurred = gray.copy()

    # 1. Sobel Operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    sobel_mag_norm = np.clip((sobel_mag / (sobel_mag.max() + 1e-8)) * 255, 0, 255).astype(np.uint8)
    sobel_angle = np.arctan2(sobel_y, sobel_x)  # Rentang [-pi, pi]

    # 2. Prewitt Operator (Konvolusi kernel diferensial)
    kernel_px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(blurred.astype(np.float32), cv2.CV_32F, kernel_px)
    prewitt_y = cv2.filter2D(blurred.astype(np.float32), cv2.CV_32F, kernel_py)
    prewitt_mag = np.hypot(prewitt_x, prewitt_y)
    prewitt_mag_norm = np.clip((prewitt_mag / (prewitt_mag.max() + 1e-8)) * 255, 0, 255).astype(np.uint8)

    # 3. Scharr Operator (Optimal untuk diferensiasi turunan orde 1 ukuran 3x3)
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr_mag = np.hypot(scharr_x, scharr_y)
    scharr_mag_norm = np.clip((scharr_mag / (scharr_mag.max() + 1e-8)) * 255, 0, 255).astype(np.uint8)

    # 4. Laplacian of Gaussian (LoG)
    log_blurred = cv2.GaussianBlur(gray, (blur_ksize if blur_ksize > 1 else 3, blur_ksize if blur_ksize > 1 else 3), 0)
    laplacian = cv2.Laplacian(log_blurred, cv2.CV_64F)
    laplacian_abs = np.clip(np.abs(laplacian) / (np.max(np.abs(laplacian)) + 1e-8) * 255, 0, 255).astype(np.uint8)

    # 5. Canny Edge Detector
    canny_edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Hitung metrik kepadatan tepi (edge density)
    total_pixels = gray.size
    edge_densities = {
        "sobel": float(np.count_nonzero(sobel_mag_norm > 50) / total_pixels),
        "prewitt": float(np.count_nonzero(prewitt_mag_norm > 50) / total_pixels),
        "scharr": float(np.count_nonzero(scharr_mag_norm > 50) / total_pixels),
        "laplacian": float(np.count_nonzero(laplacian_abs > 50) / total_pixels),
        "canny": float(np.count_nonzero(canny_edges > 0) / total_pixels),
    }

    return {
        "gray": gray,
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "sobel_mag": sobel_mag_norm,
        "sobel_angle": sobel_angle,
        "prewitt_mag": prewitt_mag_norm,
        "scharr_mag": scharr_mag_norm,
        "log_mag": laplacian_abs,
        "canny": canny_edges,
        "edge_densities": edge_densities,
    }


def visualize_edge_features(
    image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    blur_ksize: int = 3,
    figsize: Tuple[int, int] = (15, 9),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Memvisualisasikan perbandingan metode deteksi tepi dan peta gradien.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        low_thresh: Ambang histeresis bawah Canny.
        high_thresh: Ambang histeresis atas Canny.
        blur_ksize: Ukuran kernel filter Gaussian.
        figsize: Dimensi figure matplotlib.
        show: Jika True, langsung menampilkan figure dengan plt.show().
        save_path: Path untuk menyimpan file visualisasi (opsional).

    Returns:
        Dict[str, Any]: Data hasil ekstraksi tepi dan figure matplotlib.
    """
    results = compute_edge_maps(
        image,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        blur_ksize=blur_ksize
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    titles = [
        "1. Citra Asli (Grayscale)",
        f"2. Sobel Magnitude (Density: {results['edge_densities']['sobel']:.3f})",
        f"3. Prewitt Magnitude (Density: {results['edge_densities']['prewitt']:.3f})",
        f"4. Scharr Magnitude (Density: {results['edge_densities']['scharr']:.3f})",
        f"5. Laplacian of Gaussian (Density: {results['edge_densities']['laplacian']:.3f})",
        f"6. Canny Edge ({low_thresh}, {high_thresh}) (Density: {results['edge_densities']['canny']:.3f})"
    ]

    images_to_show = [
        results["gray"],
        results["sobel_mag"],
        results["prewitt_mag"],
        results["scharr_mag"],
        results["log_mag"],
        results["canny"]
    ]

    cmaps = ["gray", "inferno", "inferno", "inferno", "magma", "gray"]

    for ax, img, title, cmap in zip(axes, images_to_show, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Feature Extraction: Edge Detection & Spatial Gradients", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    results["fig"] = fig
    return results
