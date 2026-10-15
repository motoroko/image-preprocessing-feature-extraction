"""
Section 9 — Segmentasi Citra Eksploratif (Image Segmentation).

Mendukung:
- Threshold-based Segmentation (Segmentasi foreground-background dengan mask)
- K-Means Color Clustering Segmentation (Pengelompokan piksel berdasarkan kesamaan warna RGB)
- Contour Extraction dari citra biner (Visualisasi kontur objek)
- Watershed Segmentation (Segmentasi berbasis penanda topografi gradien)
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def segment_by_threshold(image: np.ndarray, thresh_val: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Segmentasi biner sederhana menghasilkan mask dan citra foreground yang terisolasi."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented


def segment_kmeans(
    image: np.ndarray,
    n_clusters: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmentasi warna menggunakan K-Means clustering.
    Mengembalikan citra terkuantisasi dan peta label region.
    """
    # Reshape ke (N, 3)
    data = image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    # Set seed untuk reproducibility
    np.random.seed(seed)
    compactness, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized_image = res.reshape(image.shape)
    label_map = labels.reshape(image.shape[:2])

    return quantized_image, label_map


def extract_contours(image: np.ndarray) -> np.ndarray:
    """Mengekstraksi dan menggambar kontur batas objek di atas citra asli."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    return contour_img


def segment_watershed(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmentasi Watershed menggunakan distance transform dan penanda (markers).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform untuk sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)

    watershed_vis = image.copy()
    watershed_vis[markers == -1] = [255, 0, 0]  # Garis batas merah

    return watershed_vis, markers


def visualize_segmentation(
    image: np.ndarray,
    n_clusters: int = 4,
    seed: int = 42,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 3,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk mengeksplorasi teknik segmentasi citra.
    """
    thresh_mask, thresh_fg = segment_by_threshold(image, thresh_val=110)
    kmeans_img, kmeans_labels = segment_kmeans(image, n_clusters=n_clusters, seed=seed)
    contour_vis = extract_contours(image)
    watershed_vis, markers = segment_watershed(image)

    results: Dict[str, np.ndarray] = {
        "original": image.copy(),
        "threshold_mask": thresh_mask,
        "threshold_foreground": thresh_fg,
        "kmeans_quantized": kmeans_img,
        "kmeans_labels": kmeans_labels,
        "contours": contour_vis,
        "watershed": watershed_vis
    }

    display_images = [
        image,
        thresh_mask,
        thresh_fg,
        kmeans_img,
        kmeans_labels,
        contour_vis,
        watershed_vis
    ]

    titles = [
        "1. Original Image",
        "2. Binary Threshold Mask",
        "3. Threshold Foreground Segment",
        f"4. K-Means Colors (k={n_clusters})",
        f"5. K-Means Cluster Labels (k={n_clusters})",
        "6. Binary Contours Overlay",
        "7. Watershed Boundaries (Red)"
    ]

    cmaps = [None, "gray", None, None, "viridis", None, None]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize if figsize else (13, 8),
        cmaps=cmaps,
        main_title="Section 9 — Eksplorasi Segmentasi Citra",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
