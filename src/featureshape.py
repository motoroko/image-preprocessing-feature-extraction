"""
Modul Ekstraksi Fitur Bentuk dan Geometri (Shape & Geometry Features).

Mengimplementasikan analisis bentuk global dan lokal:
- Contour Descriptors (Luas, Keliling, Compactness/Circularity, Bounding Box, Solidity)
- Hu Moments (7 momen invarian terhadap translasi, rotasi, dan skala)
- Hough Transform (Deteksi garis struktural & lingkaran)
- HOG (Histogram of Oriented Gradients: medan arah gradien spasial)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


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


def compute_shape_features(
    image: np.ndarray,
    min_contour_area: float = 100.0,
    hog_orientations: int = 8,
    hog_pixels_per_cell: Tuple[int, int] = (16, 16)
) -> Dict[str, Any]:
    """
    Mengekstrak deskriptor bentuk, momen Hu, Hough transform, dan HOG.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        min_contour_area: Luas minimum kontur untuk dipertimbangkan sebagai objek utama.
        hog_orientations: Jumlah bin sudut orientasi HOG.
        hog_pixels_per_cell: Ukuran sel spasial HOG dalam piksel.

    Returns:
        Dict[str, Any]: Properti kontur, Hu Moments, garis/lingkaran Hough, dan vektor HOG.
    """
    gray = _to_gray_uint8(image)
    h, w = gray.shape

    # Preprocessing biner (Otsu Thresholding + Morfologi Closing)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 1. Kontur & Deskriptor Geometris
    contours, hierarchy = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_contour = None
    contour_props = {
        "area": 0.0,
        "perimeter": 0.0,
        "circularity": 0.0,
        "aspect_ratio": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
    }
    
    if contours:
        # Ambil kontur dengan area terbesar
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        if valid_contours:
            main_contour = max(valid_contours, key=cv2.contourArea)
            area = float(cv2.contourArea(main_contour))
            perimeter = float(cv2.arcLength(main_contour, True))
            
            # Compactness / Circularity: 4 * pi * Area / Perimeter^2 (1.0 untuk lingkaran sempurna)
            circularity = float(4 * np.pi * area / (perimeter ** 2 + 1e-8))
            
            # Bounding Box & Aspect Ratio
            bx, by, bw, bh = cv2.boundingRect(main_contour)
            aspect_ratio = float(bw / (bh + 1e-8))
            extent = float(area / (bw * bh + 1e-8))
            
            # Convex Hull & Solidity
            hull = cv2.convexHull(main_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area / (hull_area + 1e-8))

            contour_props = {
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "extent": extent,
                "bounding_box": (bx, by, bw, bh)
            }

    # 2. Hu Moments (7 momen invarian skala, translasi, rotasi)
    moments = cv2.moments(binary_clean)
    raw_hu = cv2.HuMoments(moments).flatten()
    # Log scale transform: -sign(h) * log10(abs(h)) untuk stabilitas numerik
    hu_moments_log = []
    for h_val in raw_hu:
        if abs(h_val) > 1e-12:
            hu_moments_log.append(float(-np.sign(h_val) * np.log10(abs(h_val))))
        else:
            hu_moments_log.append(0.0)

    # 3. Hough Transform: Deteksi Garis & Lingkaran
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=20, maxLineGap=10)
    line_list = []
    if lines is not None:
        for line in lines[:20]:
            line_list.append(line[0].tolist())

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=int(min(h, w) * 0.4)
    )
    circle_list = []
    if circles is not None:
        circles_round = np.uint16(np.around(circles[0]))
        for c in circles_round[:10]:
            circle_list.append((int(c[0]), int(c[1]), int(c[2])))

    # 4. HOG (Histogram of Oriented Gradients)
    hog_vector, hog_image = hog(
        gray,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=(1, 1),
        visualize=True
    )

    return {
        "gray": gray,
        "binary_mask": binary_clean,
        "main_contour": main_contour,
        "contour_props": contour_props,
        "hu_moments_log": np.array(hu_moments_log, dtype=np.float32),
        "hough_lines": line_list,
        "hough_circles": circle_list,
        "hog_image": hog_image,
        "hog_vector": hog_vector,
    }


def visualize_shape_features(
    image: np.ndarray,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Memvisualisasikan analisis bentuk, kontur, Hough Transform, dan medan panah HOG.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        figsize: Dimensi figure matplotlib.
        show: Jika True, langsung menampilkan figure dengan plt.show().
        save_path: Path penyimpanan figure (opsional).

    Returns:
        Dict[str, Any]: Data hasil analisis bentuk dan figure matplotlib.
    """
    res = compute_shape_features(image)
    gray = res["gray"]
    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 1. Overlay Kontur & Convex Hull
    img_contour = base_rgb.copy()
    if res["main_contour"] is not None:
        cv2.drawContours(img_contour, [res["main_contour"]], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(res["main_contour"])
        cv2.drawContours(img_contour, [hull], -1, (255, 0, 0), 1)
        bx, by, bw, bh = res["contour_props"]["bounding_box"]
        cv2.rectangle(img_contour, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

    # 2. Overlay Hough Lines & Circles
    img_hough = base_rgb.copy()
    for x1, y1, x2, y2 in res["hough_lines"]:
        cv2.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for cx, cy, cr in res["hough_circles"]:
        cv2.circle(img_hough, (cx, cy), cr, (0, 255, 255), 2)
        cv2.circle(img_hough, (cx, cy), 2, (0, 0, 255), 3)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Subplot 0: Masker Biner Preprocessing
    axes[0].imshow(res["binary_mask"], cmap="gray")
    axes[0].set_title("1. Biner Masker (Otsu + Closing)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Subplot 1: Kontur, Convex Hull, & Bounding Box
    axes[1].imshow(img_contour)
    cp = res["contour_props"]
    axes[1].set_title(
        f"2. Kontur & Convex Hull\nCircularity={cp['circularity']:.2f}, Solidity={cp['solidity']:.2f}",
        fontsize=10, fontweight="bold"
    )
    axes[1].axis("off")

    # Subplot 2: Hough Lines & Circles
    axes[2].imshow(img_hough)
    axes[2].set_title(
        f"3. Hough Transform\n({len(res['hough_lines'])} Lines, {len(res['hough_circles'])} Circles)",
        fontsize=10, fontweight="bold"
    )
    axes[2].axis("off")

    # Subplot 3: 7 Invariant Hu Moments
    axes[3].bar(range(1, 8), res["hu_moments_log"], color="mediumpurple", width=0.6)
    axes[3].set_title("4. 7 Hu Invariant Moments (-sign*log10)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Momen Hu (h1 - h7)", fontsize=9)
    axes[3].set_ylabel("Nilai Logaritmik", fontsize=9)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: HOG Visualization
    axes[4].imshow(res["hog_image"], cmap="inferno")
    axes[4].set_title(f"5. Visualisasi HOG (Oriented Gradients)\nVektor Dimensi: {len(res['hog_vector'])}D", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    # Subplot 5: Deskriptor Metrik Geometris
    axes[5].axis("off")
    summary_text = (
        "Rangkuman Fitur Bentuk:\n"
        "-------------------------------\n"
        f"- Luas Area (Pixels): {cp['area']:.1f}\n"
        f"- Keliling (Perimeter): {cp['perimeter']:.1f}\n"
        f"- Circularity (Kerapatan): {cp['circularity']:.3f}\n"
        f"- Solidity: {cp['solidity']:.3f}\n"
        f"- Aspect Ratio (W/H): {cp['aspect_ratio']:.3f}\n"
        f"- Extent: {cp['extent']:.3f}\n"
        f"- Hough Lines Terdeteksi: {len(res['hough_lines'])}\n"
        f"- Hough Circles Terdeteksi: {len(res['hough_circles'])}\n"
        f"- Panjang Vektor HOG: {len(res['hog_vector'])} fitur\n"
        f"- Hu Momen h1 (Penyebaran): {res['hu_moments_log'][0]:.3f}"
    )
    axes[5].text(0.1, 0.5, summary_text, fontsize=9.5, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="lightgray"))

    plt.suptitle("Feature Extraction: Shape, Hu Moments, Hough, & HOG", fontsize=14, y=0.98)
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
