"""
Section 7 — Pemrosesan Noise dan Denoising (Noise Processing).

Mendukung:
- Pembuatan Noise Sintetis:
  * Gaussian Noise (Zero-mean additive noise)
  * Salt-and-Pepper Noise (Impulse noise acak)
  * Speckle Noise (Multiplicative noise)
- Denoising Filters:
  * Gaussian Filter
  * Median Filter
  * Bilateral Filter
- Metrik Evaluasi Kualitas Denoising:
  * Mean Squared Error (MSE)
  * Peak Signal-to-Noise Ratio (PSNR)
- Reproducibility menggunakan NumPy random generator dengan seed.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = 42
) -> np.ndarray:
    """Menambahkan additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    gauss = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.04,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """Menambahkan salt-and-pepper (impulse) noise."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()

    # Salt (putih)
    num_salt = int(amount * image.size * salt_vs_pepper)
    if image.ndim == 2:
        coords = [rng.integers(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[tuple(coords)] = 255
    else:
        coords = (
            rng.integers(0, image.shape[0], int(num_salt)),
            rng.integers(0, image.shape[1], int(num_salt))
        )
        noisy[coords[0], coords[1], :] = 255

    # Pepper (hitam)
    num_pepper = int(amount * image.size * (1.0 - salt_vs_pepper))
    if image.ndim == 2:
        coords = [rng.integers(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[tuple(coords)] = 0
    else:
        coords = (
            rng.integers(0, image.shape[0], int(num_pepper)),
            rng.integers(0, image.shape[1], int(num_pepper))
        )
        noisy[coords[0], coords[1], :] = 0

    return noisy


def add_speckle_noise(
    image: np.ndarray,
    var: float = 0.04,
    seed: Optional[int] = 42
) -> np.ndarray:
    """Menambahkan multiplicative speckle noise: I_noisy = I + I * noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(var), image.shape)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def compute_metrics(original: np.ndarray, test_image: np.ndarray) -> Dict[str, float]:
    """Menghitung MSE dan PSNR antara citra uji terhadap citra asli sebagai ground truth."""
    mse = float(np.mean((original.astype(np.float64) - test_image.astype(np.float64)) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        max_pixel = 255.0
        psnr = float(20.0 * np.log10(max_pixel / np.sqrt(mse)))
    return {"mse": round(mse, 2), "psnr": round(psnr, 2)}


def visualize_noise_processing(
    image: np.ndarray,
    noise_type: str = "salt_and_pepper",
    noise_intensity: float = 0.05,
    seed: int = 42,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk menyimulasikan noise dan mengevaluasi kinerja berbagai filter denoising.
    """
    ntype = noise_type.lower()
    if "gauss" in ntype:
        noisy = add_gaussian_noise(image, sigma=noise_intensity * 255.0, seed=seed)
        noise_label = f"Gaussian (σ={int(noise_intensity*255)})"
    elif "salt" in ntype or "pepper" in ntype:
        noisy = add_salt_and_pepper_noise(image, amount=noise_intensity, seed=seed)
        noise_label = f"Salt & Pepper ({int(noise_intensity*100)}%)"
    elif "speckle" in ntype:
        noisy = add_speckle_noise(image, var=noise_intensity, seed=seed)
        noise_label = f"Speckle (var={noise_intensity})"
    else:
        raise ValueError(f"Tipe noise '{noise_type}' tidak didukung.")

    # Denoising
    denoise_gauss = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)
    denoise_median = cv2.medianBlur(noisy, 5)
    denoise_bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

    # Metrik
    m_noisy = compute_metrics(image, noisy)
    m_gauss = compute_metrics(image, denoise_gauss)
    m_median = compute_metrics(image, denoise_median)
    m_bilateral = compute_metrics(image, denoise_bilateral)

    display_images = [image, noisy, denoise_gauss, denoise_median, denoise_bilateral]
    titles = [
        "1. Original (Ground Truth)",
        f"2. Noisy: {noise_label}\nPSNR: {m_noisy['psnr']} dB",
        f"3. Denoised: Gaussian Blur\nPSNR: {m_gauss['psnr']} dB",
        f"4. Denoised: Median Filter\nPSNR: {m_median['psnr']} dB",
        f"5. Denoised: Bilateral Filter\nPSNR: {m_bilateral['psnr']} dB"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=5,
        figsize=figsize if figsize else (16, 4.5),
        main_title="Section 7 — Simulasi Noise dan Evaluasi Filter Denoising",
        show=show
    )

    return {
        "results": {
            "original": image.copy(),
            "noisy": noisy,
            "gaussian_denoise": denoise_gauss,
            "median_denoise": denoise_median,
            "bilateral_denoise": denoise_bilateral
        },
        "metrics": {
            "noisy": m_noisy,
            "gaussian": m_gauss,
            "median": m_median,
            "bilateral": m_bilateral
        },
        "figure": fig
    }
