"""
Section 2 — Transformasi Geometris (Geometric Transformations).

Mendukung transformasi citra:
- Crop (koordinat ymin:ymax, xmin:xmax)
- Resize (interpolasi nearest, linear, cubic, area, lanczos)
- Rotation (derajat, pusat rotasi, border handling)
- Translation / Shift (dx, dy)
- Zoom In / Zoom Out (mempertahankan dimensi canvas atau dimensi keluaran fleksibel)
- Flip Horizontal & Flip Vertikal
- Shear (kemiringan horizontal/vertikal)
- Affine Transformation (pemetaan 3 titik)
- Perspective Transformation (pemetaan 4 titik)
- Padding & Canvas Expansion
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def crop_image(
    image: np.ndarray,
    crop_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Melakukan cropping citra.
    Koordinat crop_box dinyatakan dalam urutan: (ymin, ymax, xmin, xmax).
    """
    ymin, ymax, xmin, xmax = crop_box
    h, w = image.shape[:2]
    ymin = max(0, min(ymin, h - 1))
    ymax = max(ymin + 1, min(ymax, h))
    xmin = max(0, min(xmin, w - 1))
    xmax = max(xmin + 1, min(xmax, w))
    return image[ymin:ymax, xmin:xmax].copy()


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = "linear"
) -> np.ndarray:
    """
    Mengubah ukuran citra ke (width, height).
    """
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    flag = interp_map.get(interpolation.lower(), cv2.INTER_LINEAR)
    return cv2.resize(image, target_size, interpolation=flag)


def rotate_image(
    image: np.ndarray,
    angle: float,
    scale: float = 1.0,
    center: Optional[Tuple[float, float]] = None,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> np.ndarray:
    """
    Memutar citra dengan sudut dalam derajat berlawanan jarum jam.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)


def translate_image(
    image: np.ndarray,
    tx: float,
    ty: float,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> np.ndarray:
    """
    Menggeser citra sejauh tx piksel ke kanan dan ty piksel ke bawah.
    """
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)


def flip_image(image: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """
    Membalik citra horizontal (mode='horizontal'), vertikal (mode='vertical'), atau keduanya ('both').
    """
    if mode == "horizontal":
        return cv2.flip(image, 1)
    elif mode == "vertical":
        return cv2.flip(image, 0)
    elif mode == "both":
        return cv2.flip(image, -1)
    else:
        raise ValueError("Mode flip harus 'horizontal', 'vertical', atau 'both'.")


def zoom_image(
    image: np.ndarray,
    zoom_factor: float,
    keep_canvas_size: bool = True
) -> np.ndarray:
    """
    Melakukan zoom-in (zoom_factor > 1) atau zoom-out (zoom_factor < 1).
    Jika keep_canvas_size=True, citra hasil dipad atau dicrop agar dimensinya tetap sama dengan citra asli.
    """
    h, w = image.shape[:2]
    new_w = max(1, int(w * zoom_factor))
    new_h = max(1, int(h * zoom_factor))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if not keep_canvas_size:
        return resized

    if zoom_factor > 1.0:
        # Crop bagian tengah
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w].copy()
    else:
        # Pad dengan border hitam di sekeliling
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


def shear_image(
    image: np.ndarray,
    shear_x: float = 0.2,
    shear_y: float = 0.0
) -> np.ndarray:
    """
    Menerapkan transformasi shear (kemiringan) pada sumbu x dan/atau y.
    """
    h, w = image.shape[:2]
    matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    new_w = int(w + abs(shear_x) * h)
    new_h = int(h + abs(shear_y) * w)
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def affine_transform_demo(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan transformasi affine berdasarkan pemetaan 3 titik sudut segitiga.
    """
    h, w = image.shape[:2]
    pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])
    pts2 = np.float32([[20, 70], [w - 80, 20], [80, h - 30]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def perspective_transform_demo(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan transformasi perspektif (homografi) berdasarkan pemetaan 4 titik sudut.
    """
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([[w * 0.15, h * 0.1], [w * 0.85, h * 0.2], [0, h - 1], [w - 1, h * 0.9]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def pad_image(
    image: np.ndarray,
    pad_width: int = 30,
    border_mode: str = "constant"
) -> np.ndarray:
    """
    Menambahkan padding pada sekeliling citra.
    Pilihan border_mode: 'constant', 'reflect', 'replicate'.
    """
    mode_map = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE
    }
    flag = mode_map.get(border_mode.lower(), cv2.BORDER_CONSTANT)
    return cv2.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width, flag, value=0)


def visualize_geometric_transforms(
    image: np.ndarray,
    crop_coords: Optional[Tuple[int, int, int, int]] = None,
    rotation_angle: float = 30.0,
    shift: Tuple[float, float] = (30.0, -20.0),
    zoom_in_factor: float = 1.35,
    zoom_out_factor: float = 0.75,
    shear_factor: float = 0.2,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Any]:
    """
    Fungsi publik utama untuk mengeksplorasi dan memvisualisasikan berbagai transformasi geometris.
    """
    h, w = image.shape[:2]
    if crop_coords is None:
        crop_coords = (int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8))

    results: Dict[str, np.ndarray] = {
        "original": image.copy(),
        "crop": crop_image(image, crop_coords),
        "rotate": rotate_image(image, angle=rotation_angle),
        "translate": translate_image(image, tx=shift[0], ty=shift[1]),
        "flip_horizontal": flip_image(image, mode="horizontal"),
        "flip_vertical": flip_image(image, mode="vertical"),
        "zoom_in": zoom_image(image, zoom_factor=zoom_in_factor, keep_canvas_size=True),
        "zoom_out": zoom_image(image, zoom_factor=zoom_out_factor, keep_canvas_size=True),
        "shear": shear_image(image, shear_x=shear_factor, shear_y=0.0),
        "affine": affine_transform_demo(image),
        "perspective": perspective_transform_demo(image),
        "padded": pad_image(image, pad_width=30, border_mode="reflect"),
    }

    display_images = list(results.values())
    titles = [
        "1. Original",
        f"2. Crop ({crop_coords[0]}:{crop_coords[1]}, {crop_coords[2]}:{crop_coords[3]})",
        f"3. Rotation ({rotation_angle}°)",
        f"4. Shift dx={shift[0]}, dy={shift[1]}",
        "5. Flip Horizontal",
        "6. Flip Vertical",
        f"7. Zoom In ({zoom_in_factor}x canvas)",
        f"8. Zoom Out ({zoom_out_factor}x canvas)",
        f"9. Shear (x={shear_factor})",
        "10. Affine 3-point Warping",
        "11. Perspective Homography",
        "12. Border Padding (Reflect 30px)"
    ]

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        main_title="Section 2 — Transformasi Geometris Citra Digital",
        show=show
    )

    return {
        "results": results,
        "figure": fig
    }
