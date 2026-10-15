"""
Test suite komprehensif untuk modul filtering, histogram, enhancement, morfologi, noise, thresholding, segmentasi, piramida, numerik, dan augmentasi.
"""

import numpy as np

from src.augmentation import augment_pipeline_combined, visualize_augmentations
from src.dataloader import create_synthetic_face
from src.histogramprocessing import apply_clahe, apply_histogram_equalization
from src.imageenhancement import adjust_brightness, adjust_contrast, adjust_gamma
from src.imagefiltering import apply_box_blur, apply_canny, apply_gaussian_blur, apply_median_filter
from src.imagepyramids import build_gaussian_pyramid, build_laplacian_pyramid
from src.morphologicaloperations import apply_morphology
from src.noiseprocessing import add_gaussian_noise, compute_metrics
from src.numericalpreprocessing import normalize_to_0_1, standardize_zscore
from src.segmentation import segment_kmeans
from src.thresholding import apply_otsu_threshold


def test_filtering_immutability():
    img = create_synthetic_face(height=40, width=40)
    orig_copy = img.copy()
    _ = apply_gaussian_blur(img, kernel_size=5, sigma=1.0)
    _ = apply_box_blur(img, kernel_size=3)
    _ = apply_median_filter(img, kernel_size=3)
    _ = apply_canny(img)
    # Cek bahwa citra asli tidak dimodifikasi in-place
    assert np.array_equal(img, orig_copy)


def test_histogram_processing():
    img = create_synthetic_face(height=50, width=50)
    eq = apply_histogram_equalization(img)
    assert eq.shape == img.shape
    clahe = apply_clahe(img, clip_limit=2.0)
    assert clahe.shape == img.shape


def test_enhancement_safety():
    img = create_synthetic_face(height=40, width=40)
    b_up = adjust_brightness(img, beta=50)
    assert b_up.dtype == np.uint8
    assert b_up.max() <= 255

    g = adjust_gamma(img, gamma=1.5)
    assert g.shape == img.shape

    c = adjust_contrast(img, alpha=1.2)
    assert c.shape == img.shape


def test_morphology():
    img = create_synthetic_face(height=50, width=50)
    gray = img[:, :, 0]
    eroded = apply_morphology(gray, "erosion", kernel_size=3)
    dilated = apply_morphology(gray, "dilation", kernel_size=3)
    assert eroded.shape == gray.shape
    assert dilated.shape == gray.shape


def test_noise_and_metrics():
    img = create_synthetic_face(height=50, width=50)
    noisy1 = add_gaussian_noise(img, sigma=20.0, seed=123)
    noisy2 = add_gaussian_noise(img, sigma=20.0, seed=123)
    # Reproducible seed
    assert np.array_equal(noisy1, noisy2)

    metrics = compute_metrics(img, noisy1)
    assert metrics["mse"] > 0
    assert metrics["psnr"] > 0


def test_thresholding_otsu():
    img = create_synthetic_face(height=50, width=50)
    gray = img[:, :, 0]
    val, thresh = apply_otsu_threshold(gray)
    assert 0 <= val <= 255
    assert thresh.shape == gray.shape
    assert set(np.unique(thresh)).issubset({0, 255})


def test_segmentation_kmeans():
    img = create_synthetic_face(height=30, width=30)
    quant, labels = segment_kmeans(img, n_clusters=3, seed=42)
    assert quant.shape == img.shape
    assert labels.shape == (30, 30)
    assert len(np.unique(labels)) <= 3


def test_pyramids():
    img = create_synthetic_face(height=64, width=64)
    g_pyr = build_gaussian_pyramid(img, levels=3)
    assert len(g_pyr) == 3
    assert g_pyr[0].shape == (64, 64, 3)
    assert g_pyr[1].shape == (32, 32, 3)
    assert g_pyr[2].shape == (16, 16, 3)

    l_pyr = build_laplacian_pyramid(img, levels=3)
    assert len(l_pyr) == 3


def test_numerical_zero_division():
    # Citra konstan semua piksel bernilai sama
    const_img = np.full((30, 30), 100, dtype=np.uint8)
    norm = normalize_to_0_1(const_img)
    assert norm.shape == (30, 30)
    assert not np.isnan(norm).any()
    assert norm.max() == 0.0

    z, mean, std = standardize_zscore(const_img)
    assert not np.isnan(z).any()
    assert std == 0.0


def test_augmentation_reproducibility():
    img = create_synthetic_face(height=40, width=40)
    aug1 = augment_pipeline_combined(img, seed=99)
    aug2 = augment_pipeline_combined(img, seed=99)
    assert np.array_equal(aug1, aug2)
