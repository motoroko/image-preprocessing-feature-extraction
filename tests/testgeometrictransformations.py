"""
Test suite untuk transformasi geometris citra.
"""

import numpy as np

from src.dataloader import create_synthetic_face
from src.geometrictransformations import (
    crop_image,
    flip_image,
    resize_image,
    rotate_image,
    translate_image,
    visualize_geometric_transforms,
    zoom_image,
)


def test_crop_image():
    img = create_synthetic_face(height=100, width=100)
    cropped = crop_image(img, (20, 60, 30, 80))
    assert cropped.shape == (40, 50, 3)


def test_resize_image():
    img = create_synthetic_face(height=100, width=100)
    resized = resize_image(img, (50, 70), interpolation="linear")
    assert resized.shape == (70, 50, 3)


def test_rotate_and_translate_keep_shape():
    img = create_synthetic_face(height=64, width=64)
    rot = rotate_image(img, 45.0)
    assert rot.shape == (64, 64, 3)

    trans = translate_image(img, 10, -5)
    assert trans.shape == (64, 64, 3)


def test_flip_modes():
    img = create_synthetic_face(height=40, width=40)
    assert flip_image(img, "horizontal").shape == (40, 40, 3)
    assert flip_image(img, "vertical").shape == (40, 40, 3)


def test_zoom_canvas_preservation():
    img = create_synthetic_face(height=50, width=50)
    z_in = zoom_image(img, 1.5, keep_canvas_size=True)
    assert z_in.shape == (50, 50, 3)

    z_out = zoom_image(img, 0.7, keep_canvas_size=True)
    assert z_out.shape == (50, 50, 3)


def test_visualize_geometric_headless():
    img = create_synthetic_face(height=40, width=40)
    res = visualize_geometric_transforms(img, show=False)
    assert "results" in res
    assert "figure" in res
