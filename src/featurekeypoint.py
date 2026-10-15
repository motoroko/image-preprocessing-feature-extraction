"""
Modul Deteksi Sudut dan Titik Kunci (Corner & Keypoint Detection).

Mengimplementasikan metode deteksi titik jangkar dan deskriptor lokal invarian:
- Harris Corner Detector (Autokorelasi gradien & respons determinan-trace)
- Shi-Tomasi Corner Detector (Minimum eigenvalue score)
- FAST (Features from Accelerated Segment Test)
- SIFT (Scale-Invariant Feature Transform: detektor DoG & orientasi)
- ORB (Oriented FAST and Rotated BRIEF: detektor & deskriptor biner cepat)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def compute_keypoints(
    image: np.ndarray,
    max_corners: int = 100,
    harris_k: float = 0.04,
    harris_thresh_ratio: float = 0.01,
    fast_threshold: int = 25
) -> Dict[str, Any]:
    """
    Mengekstrak titik sudut dan keypoint lokal dengan 5 algoritma berbeda.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        max_corners: Jumlah maksimum sudut untuk metode Shi-Tomasi dan ORB/SIFT.
        harris_k: Parameter empiris Harris (biasanya 0.04 - 0.06).
        harris_thresh_ratio: Rasio ambang batas respons Harris terhadap nilai maksimum.
        fast_threshold: Ambang batas intensitas diferensial untuk detector FAST.

    Returns:
        Dict[str, Any]: Koordinat titik sudut, keypoints OpenCV, dan deskriptor lokal.
    """
    gray = _to_gray_uint8(image)
    h, w = gray.shape

    # 1. Harris Corner Detection
    gray_float = np.float32(gray)
    harris_dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=harris_k)
    harris_dst_dilated = cv2.dilate(harris_dst, None)
    harris_mask = harris_dst > (harris_thresh_ratio * harris_dst.max())
    harris_coords = np.column_stack(np.where(harris_mask))  # baris (y), kolom (x)
    # Tukar ke urutan (x, y)
    harris_points = [(int(pt[1]), int(pt[0])) for pt in harris_coords[:max_corners]]

    # 2. Shi-Tomasi Corner Detection (Good Features to Track)
    corners_st = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3
    )
    shi_tomasi_points = []
    if corners_st is not None:
        shi_tomasi_points = [(int(pt[0][0]), int(pt[0][1])) for pt in corners_st]

    # 3. FAST (Features from Accelerated Segment Test)
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
    fast_kp = fast.detect(gray, None)
    if len(fast_kp) > max_corners:
        fast_kp = sorted(fast_kp, key=lambda x: x.response, reverse=True)[:max_corners]

    # 4. SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create(nfeatures=max_corners)
    sift_kp, sift_desc = sift.detectAndCompute(gray, None)

    # 5. ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create(nfeatures=max_corners)
    orb_kp, orb_desc = orb.detectAndCompute(gray, None)

    counts = {
        "harris": len(harris_points),
        "shi_tomasi": len(shi_tomasi_points),
        "fast": len(fast_kp),
        "sift": len(sift_kp) if sift_kp is not None else 0,
        "orb": len(orb_kp) if orb_kp is not None else 0,
    }

    return {
        "gray": gray,
        "harris_points": harris_points,
        "harris_response": harris_dst,
        "shi_tomasi_points": shi_tomasi_points,
        "fast_kp": fast_kp,
        "sift_kp": sift_kp,
        "sift_desc": sift_desc,
        "orb_kp": orb_kp,
        "orb_desc": orb_desc,
        "counts": counts,
    }


def visualize_keypoint_features(
    image: np.ndarray,
    max_corners: int = 100,
    figsize: Tuple[int, int] = (15, 9),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Memvisualisasikan titik sudut dan keypoints di atas citra wajah.

    Parameters:
        image: Citra input (RGB atau Grayscale).
        max_corners: Jumlah maksimum titik yang diplot.
        figsize: Dimensi figure matplotlib.
        show: Jika True, langsung menampilkan figure dengan plt.show().
        save_path: Path penyimpanan figure (opsional).

    Returns:
        Dict[str, Any]: Hasil ekstraksi dan figure matplotlib.
    """
    res = compute_keypoints(image, max_corners=max_corners)
    gray = res["gray"]
    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Buat kanvas overlay untuk masing-masing metode
    # 1. Harris
    img_harris = base_rgb.copy()
    for x, y in res["harris_points"]:
        cv2.circle(img_harris, (x, y), 3, (255, 0, 0), -1)

    # 2. Shi-Tomasi
    img_st = base_rgb.copy()
    for x, y in res["shi_tomasi_points"]:
        cv2.circle(img_st, (x, y), 3, (0, 255, 0), -1)

    # 3. FAST
    img_fast = cv2.drawKeypoints(
        base_rgb,
        res["fast_kp"],
        None,
        color=(255, 165, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # 4. SIFT (dengan lingkaran skala dan garis orientasi)
    img_sift = cv2.drawKeypoints(
        base_rgb,
        res["sift_kp"] if res["sift_kp"] else [],
        None,
        color=(0, 200, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 5. ORB (dengan lingkaran skala dan orientasi)
    img_orb = cv2.drawKeypoints(
        base_rgb,
        res["orb_kp"] if res["orb_kp"] else [],
        None,
        color=(255, 50, 150),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    titles = [
        "1. Citra Input (Grayscale)",
        f"2. Harris Corners ({res['counts']['harris']} titik)",
        f"3. Shi-Tomasi Corners ({res['counts']['shi_tomasi']} titik)",
        f"4. FAST Keypoints ({res['counts']['fast']} titik)",
        f"5. SIFT Keypoints ({res['counts']['sift']} titik, 128D Desc)",
        f"6. ORB Keypoints ({res['counts']['orb']} titik, 32B Desc)"
    ]

    displays = [base_rgb, img_harris, img_st, img_fast, img_sift, img_orb]

    for ax, img, title in zip(axes, displays, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Feature Extraction: Corner & Keypoint Detection", fontsize=14, y=0.98)
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
