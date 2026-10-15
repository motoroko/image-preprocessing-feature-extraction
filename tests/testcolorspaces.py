"""
Test suite untuk memverifikasi fungsionalitas ruang warna.
"""

import numpy as np
import pytest

from src.colorspaces import convert_color_space, rgb_to_cmyk, visualize_color_spaces
from src.dataloader import create_synthetic_face


def test_convert_color_spaces_shapes():
    img = create_synthetic_face(height=64, width=64)

    gray = convert_color_space(img, "gray")
    assert gray.shape == (64, 64)
    assert gray.ndim == 2

    hsv = convert_color_space(img, "hsv")
    assert hsv.shape == (64, 64, 3)

    lab = convert_color_space(img, "lab")
    assert lab.shape == (64, 64, 3)

    bgr = convert_color_space(img, "bgr")
    assert bgr.shape == (64, 64, 3)

    cmyk = convert_color_space(img, "cmyk")
    assert cmyk.shape == (64, 64, 4)


def test_cmyk_range():
    img = create_synthetic_face(height=32, width=32)
    cmyk = rgb_to_cmyk(img)
    assert cmyk.dtype == np.uint8
    assert cmyk.min() >= 0
    assert cmyk.max() <= 255


def test_invalid_space():
    img = create_synthetic_face(height=32, width=32)
    with pytest.raises(ValueError):
        convert_color_space(img, "invalid_space_xyz")


def test_visualize_color_spaces_headless():
    img = create_synthetic_face(height=48, width=48)
    res = visualize_color_spaces(img, spaces=["rgb", "gray", "hsv"], show_channels=True, show=False)
    assert "conversions" in res
    assert "figure" in res
