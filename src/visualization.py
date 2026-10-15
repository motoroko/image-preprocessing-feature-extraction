"""
Modul Utilitas Visualisasi Bersama.

Menyediakan fungsi standar untuk plotting grid subplot citra, formatting judul,
colorbar, histogram, serta penyimpanan figure secara konsisten di seluruh modul.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    cmaps: Optional[Union[str, List[Optional[str]]]] = None,
    main_title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    subtitles_fontsize: int = 10
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Menampilkan daftar gambar dalam grid subplot Matplotlib secara rapi dan konsisten.

    Parameters:
        images: List dari array citra NumPy (2D grayscale atau 3D RGB/berwarna).
        titles: List judul untuk tiap subplot.
        ncols: Jumlah kolom grid visualisasi.
        figsize: Ukuran figure (width, height) dalam inci. Jika None, dihitung otomatis.
        cmaps: Colormap string atau list colormap untuk masing-masing subplot.
        main_title: Judul utama figure keseluruhan (suptitle).
        show: Jika True, memanggil plt.show().
        save_path: Jika diberikan, menyimpan figure ke file path ini.
        subtitles_fontsize: Ukuran font judul subplot.

    Returns:
        Tuple[plt.Figure, np.ndarray]: Objek Figure dan Axes Matplotlib.
    """
    n_images = len(images)
    if n_images == 0:
        raise ValueError("List images tidak boleh kosong.")

    nrows = int(np.ceil(n_images / ncols))
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # Normalisasi cmaps ke list
    if cmaps is None or isinstance(cmaps, str):
        cmap_list = [cmaps] * n_images
    else:
        cmap_list = cmaps

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_images:
            img = images[i]
            title = titles[i] if titles and i < len(titles) else f"Image {i+1}"
            cmap = cmap_list[i] if i < len(cmap_list) else None

            # Pastikan visualisasi benar sesuai dimensi
            if img.ndim == 2:
                # Grayscale
                actual_cmap = cmap if cmap is not None else "gray"
                # Cek jika data float atau uint8
                vmin, vmax = (0, 1) if img.dtype == np.float32 and img.max() <= 1.0 else (None, None)
                ax.imshow(img, cmap=actual_cmap, vmin=vmin, vmax=vmax)
            elif img.ndim == 3:
                # 3-channel
                if img.shape[2] == 1:
                    ax.imshow(img[:, :, 0], cmap=cmap if cmap else "gray")
                elif img.shape[2] == 3:
                    # Asumsi RGB (atau konversi float clip jika diperlukan)
                    if np.issubdtype(img.dtype, np.floating) and (img.max() > 1.0 or img.min() < 0.0):
                        disp_img = np.clip(img / 255.0 if img.max() > 1.0 else img, 0, 1)
                    else:
                        disp_img = img
                    ax.imshow(disp_img)
                elif img.shape[2] == 4:
                    # RGBA
                    ax.imshow(img)
                else:
                    ax.imshow(img[:, :, 0], cmap="gray")
            else:
                ax.imshow(np.squeeze(img), cmap="gray")

            ax.set_title(title, fontsize=subtitles_fontsize, pad=6)
            ax.axis("off")
        else:
            # Sembunyikan subplot sisa
            ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=subtitles_fontsize + 4, y=0.99, weight="bold")

    fig.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig, axes


def plot_comparison_with_histogram(
    original: np.ndarray,
    processed: np.ndarray,
    title_orig: str = "Original",
    title_proc: str = "Processed",
    bins: int = 256,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Menampilkan perbandingan citra sebelum & sesudah beserta histogram intensitasnya.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Citra Asli
    if original.ndim == 2:
        axes[0, 0].imshow(original, cmap="gray")
    else:
        axes[0, 0].imshow(original)
    axes[0, 0].set_title(title_orig)
    axes[0, 0].axis("off")

    # Histogram Asli
    if original.ndim == 2:
        axes[1, 0].hist(original.ravel(), bins=bins, range=(0, 256), color="black", alpha=0.7)
    else:
        colors = ("red", "green", "blue")
        for c_idx, color in enumerate(colors):
            axes[1, 0].hist(original[:, :, c_idx].ravel(), bins=bins, range=(0, 256), color=color, alpha=0.5, label=color.upper())
        axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].set_title(f"Histogram {title_orig}")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Citra Hasil
    if processed.ndim == 2:
        axes[0, 1].imshow(processed, cmap="gray")
    else:
        axes[0, 1].imshow(processed)
    axes[0, 1].set_title(title_proc)
    axes[0, 1].axis("off")

    # Histogram Hasil
    if processed.ndim == 2:
        axes[1, 1].hist(processed.ravel(), bins=bins, range=(0, 256), color="black", alpha=0.7)
    else:
        colors = ("red", "green", "blue")
        for c_idx, color in enumerate(colors):
            axes[1, 1].hist(processed[:, :, c_idx].ravel(), bins=bins, range=(0, 256), color=color, alpha=0.5, label=color.upper())
        axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].set_title(f"Histogram {title_proc}")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
