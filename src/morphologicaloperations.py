"""
Section 6 — Operasi Morfologi (Morphological Operations).

Mendukung operasi matematika morfologi:
- Erosion (pengikisan objek latar depan)
- Dilation (pelebaran / ekspansi objek latar depan)
- Opening (Erosion diikuti Dilation: penghapusan noise kecil / isolasi)
- Closing (Dilation diikuti Erosion: penutupan lubang kecil / penggabungan)
- Morphological Gradient (Dilation - Erosion: deteksi kontur batas)
- Top-Hat (Original - Opening: elemen terang lebih kecil dari structuring element)
- Black-Hat (Closing - Original: elemen gelap lebih kecil dari structuring element)
Dapat diterapkan pada citra biner (thresholded) maupun grayscale.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def get_structuring_element(shape: str = "rect", ksize: int = 5) -> np.ndarray:
    """
    Menghasilkan structuring element (kernel) dengan bentuk:
    - 'rect': Persegi panjang / kotak (cv2.MORPH_RECT)
    - 'ellipse': Elips / lingkaran (cv2.MORPH_ELLIPSE)
    - 'cross': Tanda silang + (cv2.MORPH_CROSS)
    """
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    flag = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.getStructuringElement(flag, (k, k))


def apply_morphology(
    image: np.ndarray,
    operation: str,
    kernel_size: int = 5,
    kernel_shape: str = "rect",
    iterations: int = 1
) -> np.ndarray:
    """
    Menerapkan operasi morfologi pada citra 2D (grayscale atau biner).
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    op = operation.lower().strip()

    if op == "erosion":
        return cv2.erode(image, kernel, iterations=iterations)
    elif op == "dilation":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif op == "opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op == "gradient":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op == "tophat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op == "blackhat":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        raise ValueError(f"Operasi morfologi '{operation}' tidak dikenali.")


def visualize_morphological_operations(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "rect",
    iterations: int = 1,
    use_binary: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk membandingkan berbagai operasi morfologi matematis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    if use_binary:
        # Binarisasi menggunakan Otsu thresholding sebagai input morfologi biner
        _, base_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        input_desc = "Binary Input (Otsu)"
    else:
        base_img = gray
        input_desc = "Grayscale Input"

    ops = [
        "erosion", "dilation", "opening", "closing",
        "gradient", "tophat", "blackhat"
    ]

    results: Dict[str, np.ndarray] = {
        "input_base": base_img.copy()
    }
    for op in ops:
        results[op] = apply_morphology(
            base_img,
            operation=op,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=iterations
        )

    display_images = list(results.values())
    titles = [
        f"1. {input_desc}",
        f"2. Erosion (k={kernel_size})",
        f"3. Dilation (k={kernel_size})",
        "4. Opening (Erode -> Dilate)",
        "5. Closing (Dilate -> Erode)",
        "6. Morph Gradient (Dilate - Erode)",
        "7. Top-Hat (Input - Open)",
        "8. Black-Hat (Close - Input)"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        cmaps="gray",
        main_title=f"Section 6 — Operasi Morfologi Citra (Shape: {kernel_shape}, Kernel: {kernel_size}x{kernel_size})",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
