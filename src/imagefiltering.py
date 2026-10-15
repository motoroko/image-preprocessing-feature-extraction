"""
Section 3 — Filtering dan Konvolusi Citra (Image Filtering & Convolution).

Mendukung operasi spasial:
- Box Blur / Average filter
- Gaussian Blur (parameter kernel dan sigma)
- Median Filter (denoising non-linear)
- Bilateral Filter (edge-preserving smoothing)
- Sharpening (Laplacian / Unsharp Masking)
- Sobel Filter (Gradien horizontal & vertikal, magnitudo)
- Laplacian Filter (Turunan kedua)
- Canny Edge Detection (Non-maximum suppression & hysteresis)
- Custom 2D Kernel Convolution
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def apply_box_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Menerapkan Box/Average blur dengan ukuran kernel ganjil."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.blur(image, (k, k))


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """Menerapkan Gaussian smoothing filter."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Menerapkan Median filter untuk menghilangkan impuls noise tanpa mengaburkan tepi berlebih."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, k)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Menerapkan edge-preserving bilateral filter."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Menerapkan sharpening menggunakan kernel unsharp mask / laplacian."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + strength, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_sobel(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menghitung gradien Sobel horizontal (dx), vertikal (dy), dan magnitude.
    Selalu diproses pada citra grayscale untuk interpretasi gradien tepi yang benar.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    k = ksize if ksize % 2 == 1 else ksize + 1
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    norm_x = cv2.convertScaleAbs(grad_x)
    norm_y = cv2.convertScaleAbs(grad_y)
    norm_mag = np.clip(magnitude / (magnitude.max() + 1e-7) * 255.0, 0, 255).astype(np.uint8)
    return norm_x, norm_y, norm_mag


def apply_laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Menghitung turunan spasial kedua (Laplacian) untuk deteksi perubahan cepat intensitas."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    k = ksize if ksize % 2 == 1 else ksize + 1
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(lap)


def apply_canny(image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """Menerapkan Canny Edge Detector bertingkat dengan non-maximum suppression."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, threshold1=low_thresh, threshold2=high_thresh)


def apply_custom_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Melakukan konvolusi 2D dengan kernel pengguna sembarang."""
    res = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)


def visualize_filters(
    image: np.ndarray,
    kernel_size: int = 5,
    gaussian_sigma: float = 1.5,
    canny_thresholds: Tuple[int, int] = (50, 150),
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan berbagai metode filtering spasial dan deteksi tepi.
    """
    sobel_x, sobel_y, sobel_mag = apply_sobel(image, ksize=3)
    laplacian = apply_laplacian(image, ksize=3)
    canny = apply_canny(image, low_thresh=canny_thresholds[0], high_thresh=canny_thresholds[1])

    # Emboss kernel sebagai contoh custom filter
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float32)

    results: Dict[str, np.ndarray] = {
        "original": image.copy(),
        "box_blur": apply_box_blur(image, kernel_size=kernel_size),
        "gaussian_blur": apply_gaussian_blur(image, kernel_size=kernel_size, sigma=gaussian_sigma),
        "median_filter": apply_median_filter(image, kernel_size=kernel_size),
        "bilateral_filter": apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75),
        "sharpened": apply_sharpening(image, strength=1.5),
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "sobel_magnitude": sobel_mag,
        "laplacian": laplacian,
        "canny": canny,
        "custom_emboss": apply_custom_kernel(image, emboss_kernel),
    }

    display_images = list(results.values())
    titles = [
        "1. Original",
        f"2. Box Blur ({kernel_size}x{kernel_size})",
        f"3. Gaussian Blur (σ={gaussian_sigma})",
        f"4. Median Filter ({kernel_size}x{kernel_size})",
        "5. Bilateral (Edge-Preserving)",
        "6. Sharpening Filter",
        "7. Sobel X (Horizontal Grad)",
        "8. Sobel Y (Vertical Grad)",
        "9. Sobel Magnitude (Total)",
        "10. Laplacian (2nd Deriv)",
        f"11. Canny ({canny_thresholds[0]},{canny_thresholds[1]})",
        "12. Custom 2D Kernel (Emboss)"
    ]

    cmaps = [
        None, None, None, None, None, None,
        "gray", "gray", "gray", "gray", "gray", "gray"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        cmaps=cmaps,
        main_title="Section 3 — Filtering Spasial, Konvolusi, dan Deteksi Tepi",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
