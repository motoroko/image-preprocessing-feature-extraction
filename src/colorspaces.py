"""
Section 1 — Representasi dan Ruang Warna (Color Spaces).

Mendukung konversi dan visualisasi ruang warna:
- RGB, BGR (urutan kanal)
- Grayscale
- HSV, HLS
- CIE LAB, XYZ
- YCrCb, YUV
- CMYK (subtraktif)
Serta ekstraksi dan visualisasi kanal individual dengan colormap yang tepat.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_image_grid


def rgb_to_cmyk(image_rgb: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra RGB uint8 ke representasi CMYK (float 0-1 atau uint8 0-255).
    OpenCV tidak menyediakan fungsi bawaan cvtColor untuk CMYK, sehingga
    konversi matematis subtraktif diimplementasikan secara eksplisit:
        K = 1 - max(R', G', B')
        C = (1 - R' - K) / (1 - K)
        M = (1 - G' - K) / (1 - K)
        Y = (1 - B' - K) / (1 - K)
    """
    rgb = image_rgb.astype(np.float32) / 255.0
    k = 1.0 - np.max(rgb, axis=2)
    # Hindari pembagian dengan nol saat K=1 (hitam pekat)
    denom = 1.0 - k
    denom[denom == 0] = 1e-7

    c = (1.0 - rgb[:, :, 0] - k) / denom
    m = (1.0 - rgb[:, :, 1] - k) / denom
    y = (1.0 - rgb[:, :, 2] - k) / denom

    c = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    m = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    k = np.clip(k * 255.0, 0, 255).astype(np.uint8)

    return np.stack([c, m, y, k], axis=-1)


def convert_color_space(
    image: np.ndarray,
    target_space: str
) -> np.ndarray:
    """
    Mengonversi citra input RGB ke ruang warna target.

    Parameters:
        image: Array NumPy RGB (H, W, 3).
        target_space: Salah satu dari ['rgb', 'bgr', 'gray', 'hsv', 'hls', 'lab', 'xyz', 'ycrcb', 'yuv', 'cmyk'].

    Returns:
        np.ndarray: Array citra dalam ruang warna target.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image harus berupa numpy.ndarray.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("convert_color_space membutuhkan citra 3-kanal RGB.")

    space = target_space.lower().strip()

    if space == "rgb":
        return image.copy()
    elif space == "bgr":
        # Perubahan urutan kanal tanpa perubahan model matematis ruang warna
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif space == "gray" or space == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif space == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif space == "hls":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif space == "lab" or space == "cie_lab":
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif space == "xyz":
        return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    elif space == "ycrcb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif space == "yuv":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif space == "cmyk":
        return rgb_to_cmyk(image)
    else:
        raise ValueError(f"Ruang warna '{target_space}' tidak didukung.")


def visualize_color_spaces(
    image: np.ndarray,
    spaces: Optional[List[str]] = None,
    show_channels: bool = True,
    channel_space: str = "hsv",
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 4,
    show: bool = True
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray], plt.Figure]]:
    """
    Fungsi publik utama untuk membandingkan ruang warna dan visualisasi kanal individual.

    Parameters:
        image: Citra input RGB (H, W, 3).
        spaces: Daftar ruang warna yang ingin dikonversi, default: ['rgb', 'bgr', 'gray', 'hsv', 'lab', 'ycrcb', 'yuv', 'cmyk'].
        show_channels: Jika True, menampilkan visualisasi kanal individual dari ruang warna `channel_space`.
        channel_space: Ruang warna untuk ditampilkan kanal individualnya (contoh: 'rgb', 'hsv', 'lab', 'cmyk').
        figsize: Ukuran figure.
        ncols: Jumlah kolom grid.
        show: Apakah figure ditampilkan dengan plt.show().

    Returns:
        Dict berisi:
            - 'conversions': Dict hasil konversi array tiap ruang warna
            - 'channels': Dict kanal individual jika show_channels=True
            - 'figure': Objek Figure Matplotlib
    """
    if spaces is None:
        spaces = ["rgb", "bgr", "gray", "hsv", "lab", "ycrcb", "yuv", "cmyk"]

    conversions: Dict[str, np.ndarray] = {}
    display_images: List[np.ndarray] = []
    titles: List[str] = []
    cmaps: List[Optional[str]] = []

    for sp in spaces:
        conv = convert_color_space(image, sp)
        conversions[sp] = conv

        if sp == "rgb":
            display_images.append(conv)
            titles.append("RGB (Original)")
            cmaps.append(None)
        elif sp == "bgr":
            # Ditampilkan langsung untuk mendemonstrasikan kanal R dan B yang tertukar
            display_images.append(conv)
            titles.append("BGR (Channel Swap)")
            cmaps.append(None)
        elif sp == "gray":
            display_images.append(conv)
            titles.append("Grayscale (Luminance)")
            cmaps.append("gray")
        elif sp == "cmyk":
            # CMYK memiliki 4 kanal, untuk preview komposit kita tampilkan kanal K (Key/Black)
            display_images.append(conv[:, :, 3])
            titles.append("CMYK (K - Black channel)")
            cmaps.append("gray")
        else:
            # Ruang warna seperti HSV, LAB, YCrCb, YUV tidak merepresentasikan RGB display langsung
            # Kita tampilkan kanal luminans/komponen utama agar representatif secara visual
            primary_ch = conv[:, :, 0]
            display_images.append(primary_ch)
            titles.append(f"{sp.upper()} (Ch 0 / Intensity)")
            cmaps.append("inferno" if sp == "hsv" else "gray")

    channels_data: Dict[str, np.ndarray] = {}
    if show_channels:
        # Ekstrak kanal individual dari channel_space
        c_space = channel_space.lower()
        if c_space not in conversions:
            conversions[c_space] = convert_color_space(image, c_space)

        raw = conversions[c_space]
        if c_space == "rgb":
            names = ["R (Red)", "G (Green)", "B (Blue)"]
            color_cmaps = ["Reds", "Greens", "Blues"]
            for idx, (nm, cm) in enumerate(zip(names, color_cmaps)):
                ch = raw[:, :, idx]
                channels_data[nm] = ch
                display_images.append(ch)
                titles.append(f"Channel: {nm}")
                cmaps.append(cm)
        elif c_space == "hsv":
            names = ["Hue (0-179)", "Saturation (0-255)", "Value (0-255)"]
            color_cmaps = ["hsv", "Blues", "gray"]
            for idx, (nm, cm) in enumerate(zip(names, color_cmaps)):
                ch = raw[:, :, idx]
                channels_data[nm] = ch
                display_images.append(ch)
                titles.append(f"HSV: {nm}")
                cmaps.append(cm)
        elif c_space == "lab":
            names = ["L (Lightness)", "a (Green-Red)", "b (Blue-Yellow)"]
            color_cmaps = ["gray", "PiYG", "YlOrBr"]
            for idx, (nm, cm) in enumerate(zip(names, color_cmaps)):
                ch = raw[:, :, idx]
                channels_data[nm] = ch
                display_images.append(ch)
                titles.append(f"LAB: {nm}")
                cmaps.append(cm)
        elif c_space == "cmyk":
            names = ["Cyan", "Magenta", "Yellow", "Black (K)"]
            color_cmaps = ["Blues", "RdPu", "YlOrBr", "gray"]
            for idx, (nm, cm) in enumerate(zip(names, color_cmaps)):
                ch = raw[:, :, idx]
                channels_data[nm] = ch
                display_images.append(ch)
                titles.append(f"CMYK: {nm}")
                cmaps.append(cm)

    fig, _ = plot_image_grid(
        images=display_images,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        cmaps=cmaps,
        main_title="Section 1 — Eksplorasi Ruang Warna & Kanal Individual",
        show=show
    )

    return {
        "conversions": conversions,
        "channels": channels_data,
        "figure": fig
    }
