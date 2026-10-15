"""
Modul Ekstraksi Fitur Tekstur (Texture Features).

Mengimplementasikan analisis tekstur statistik dan mikro:
- GLCM (Gray-Level Co-occurrence Matrix): Properti Haralick (Contrast, Dissimilarity, Homogeneity, Energy, Correlation)
- LBP (Local Binary Patterns): Tekstur biner mikro terarah dan histogram pola invarian rotasi
- Gabor Filter Bank: Filter frekuensi spasial multi-arah dan multi-skala
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Helper internal konversi citra ke grayscale uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input citra harus berupa numpy.ndarray.")
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Dimensi array citra tidak didukung (harus 2D atau 3D).")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def compute_texture_features(
    image: np.ndarray,
    glcm_distances: Tuple[int, ...] = (1, 2),
    glcm_angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lbp_points: int = 8,
    lbp_radius: float = 1.0,
    gabor_frequencies: Tuple[float, ...] = (0.1, 0.25),
    gabor_angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
) -> Dict[str, Any]:
    """
    Mengekstrak deskriptor tekstur menggunakan GLCM, LBP, dan bank filter Gabor.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        glcm_distances: Jarak ketetanggaan piksel untuk GLCM.
        glcm_angles: Sudut arah ketetanggaan (dalam radian).
        lbp_points: Jumlah piksel tetangga melingkar untuk LBP (P).
        lbp_radius: Radius lingkaran tetangga untuk LBP (R).
        gabor_frequencies: Frekuensi spasial gelombang sinusoidal Gabor.
        gabor_angles: Orientasi sudut kernel Gabor.

    Returns:
        Dict[str, Any]: Fitur Haralick GLCM, citra & histogram LBP, serta respons Gabor.
    """
    gray = _to_gray_uint8(image)

    # 1. GLCM (Gray-Level Co-occurrence Matrix)
    # Kuantisasi citra ke 16 tingkat keabuan agar GLCM efisien dan stabil
    gray_quantized = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(
        gray_quantized,
        distances=list(glcm_distances),
        angles=list(glcm_angles),
        levels=16,
        symmetric=True,
        normed=True
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    glcm_props = {}
    for prop in properties:
        val = graycoprops(glcm, prop)
        glcm_props[prop] = float(np.mean(val))
        glcm_props[f"{prop}_all"] = val.tolist()

    # 2. Local Binary Patterns (LBP)
    lbp = local_binary_pattern(gray, P=lbp_points, R=lbp_radius, method="uniform")
    n_bins = int(lbp_points * (lbp_points - 1) + 3)  # formula bins uniform LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # 3. Gabor Filter Bank
    gabor_responses = []
    gabor_features = []
    for freq in gabor_frequencies:
        for theta in gabor_angles:
            # cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi)
            wavelength = 1.0 / freq if freq > 0 else 4.0
            kernel = cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=3.0,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            response_mag = np.abs(filtered)
            gabor_responses.append({
                "freq": freq,
                "theta_deg": float(np.degrees(theta)),
                "image": np.clip(response_mag, 0, 255).astype(np.uint8),
                "mean": float(np.mean(response_mag)),
                "energy": float(np.mean(response_mag ** 2))
            })
            gabor_features.append(float(np.mean(response_mag)))
            gabor_features.append(float(np.std(response_mag)))

    return {
        "gray": gray,
        "glcm": glcm,
        "glcm_props": glcm_props,
        "lbp_image": lbp,
        "lbp_hist": lbp_hist,
        "gabor_responses": gabor_responses,
        "gabor_features": np.array(gabor_features, dtype=np.float32),
    }


def visualize_texture_features(
    image: np.ndarray,
    lbp_points: int = 8,
    lbp_radius: float = 1.0,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Memvisualisasikan analisis tekstur GLCM, citra & histogram LBP, serta respons Gabor.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        lbp_points: Jumlah tetangga LBP.
        lbp_radius: Radius LBP.
        figsize: Dimensi figure matplotlib.
        show: Jika True, langsung menampilkan figure dengan plt.show().
        save_path: Path penyimpanan figure (opsional).

    Returns:
        Dict[str, Any]: Hasil fitur tekstur dan figure matplotlib.
    """
    res = compute_texture_features(
        image,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4)

    # 1. Citra Asli
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(res["gray"], cmap="gray")
    ax0.set_title("1. Citra Grayscale", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # 2. Citra LBP
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(res["lbp_image"], cmap="magma")
    ax1.set_title(f"2. Peta LBP (P={lbp_points}, R={lbp_radius})", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # 3. Histogram LBP
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(range(len(res["lbp_hist"])), res["lbp_hist"], color="teal", width=0.8)
    ax2.set_title("3. Histogram LBP (Tekstur Mikro)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Bin Pola LBP", fontsize=8)
    ax2.set_ylabel("Frekuensi Relatif", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 4. Properti GLCM Haralick
    ax3 = fig.add_subplot(gs[0, 3])
    props_names = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    props_vals = [res["glcm_props"][p] for p in props_names]
    # Normalisasi min-max sederhana untuk visualisasi bar chart yang seimbang
    norm_vals = [(v - min(props_vals)) / (max(props_vals) - min(props_vals) + 1e-8) for v in props_vals]
    bars = ax3.barh(props_names, norm_vals, color="coral")
    ax3.set_title("4. Fitur GLCM Haralick (Skor Relatif)", fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 1.2)
    for bar, actual in zip(bars, props_vals):
        ax3.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{actual:.3f}", va="center", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Baris 2: 4 Sampel Respons Gabor Filter (berbagai orientasi sudut)
    gabor_samples = res["gabor_responses"][:4]
    for idx, sample in enumerate(gabor_samples):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(sample["image"], cmap="inferno")
        ax.set_title(f"Gabor (\u03b8={sample['theta_deg']:.0f}\u00b0, f={sample['freq']})\nMean={sample['mean']:.1f}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Feature Extraction: Texture Analysis (GLCM, LBP, & Gabor Bank)", fontsize=14, y=0.98)
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
