"""
Test suite untuk memverifikasi fungsionalitas modul-modul Feature Extraction.
"""

import numpy as np
import pytest

from src.dataloader import create_synthetic_face
from src.featureedge import compute_edge_maps, visualize_edge_features
from src.featurekeypoint import compute_keypoints, visualize_keypoint_features
from src.featuretexture import compute_texture_features, visualize_texture_features
from src.featurecolor import compute_color_features, visualize_color_features
from src.featureshape import compute_shape_features, visualize_shape_features


@pytest.fixture
def sample_image():
    return create_synthetic_face(height=128, width=128, seed=42)


def test_edge_detection(sample_image):
    res = compute_edge_maps(sample_image)
    assert "sobel_mag" in res
    assert "prewitt_mag" in res
    assert "scharr_mag" in res
    assert "log_mag" in res
    assert "canny" in res
    assert res["sobel_mag"].shape == (128, 128)
    assert res["canny"].shape == (128, 128)
    assert 0.0 <= res["edge_densities"]["canny"] <= 1.0

    vis = visualize_edge_features(sample_image, show=False)
    assert vis["fig"] is not None


def test_keypoint_detection(sample_image):
    res = compute_keypoints(sample_image, max_corners=50)
    assert "harris_points" in res
    assert "shi_tomasi_points" in res
    assert "fast_kp" in res
    assert "sift_kp" in res
    assert "orb_kp" in res
    assert isinstance(res["counts"], dict)

    vis = visualize_keypoint_features(sample_image, show=False)
    assert vis["fig"] is not None


def test_texture_features(sample_image):
    res = compute_texture_features(sample_image, lbp_points=8, lbp_radius=1.0)
    assert "glcm_props" in res
    assert "contrast" in res["glcm_props"]
    assert "homogeneity" in res["glcm_props"]
    assert res["lbp_image"].shape == (128, 128)
    assert len(res["lbp_hist"]) > 0
    assert len(res["gabor_responses"]) > 0
    assert len(res["gabor_features"]) > 0

    vis = visualize_texture_features(sample_image, show=False)
    assert vis["fig"] is not None


def test_color_features(sample_image):
    res = compute_color_features(sample_image, n_dominant=4)
    assert "moments" in res
    assert "rgb" in res["moments"]
    assert "hsv" in res["moments"]
    assert "lab" in res["moments"]
    assert res["color_vector"].shape == (27,)
    assert res["dominant_palette"].shape == (4, 3)
    assert len(res["dominant_percentages"]) == 4

    vis = visualize_color_features(sample_image, show=False)
    assert vis["fig"] is not None


def test_shape_features(sample_image):
    res = compute_shape_features(sample_image)
    assert "contour_props" in res
    assert "area" in res["contour_props"]
    assert "perimeter" in res["contour_props"]
    assert len(res["hu_moments_log"]) == 7
    assert len(res["hog_vector"]) > 0

    vis = visualize_shape_features(sample_image, show=False)
    assert vis["fig"] is not None
