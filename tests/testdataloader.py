"""
Test suite untuk memverifikasi fungsionalitas data_loader.
"""

from pathlib import Path
import numpy as np
import pytest

from src.dataloader import (
    create_synthetic_face,
    download_kaggle_dataset,
    extract_zip_files,
    get_image_list,
    get_image_metadata,
    load_experiment_image,
    load_image,
)


def test_create_synthetic_face():
    img = create_synthetic_face(height=128, width=128)
    assert isinstance(img, np.ndarray)
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_get_image_metadata():
    img = create_synthetic_face(height=100, width=120)
    meta = get_image_metadata(img)
    assert meta["height"] == 100
    assert meta["width"] == 120
    assert meta["channels"] == 3
    assert meta["dtype"] == "uint8"
    assert "mean_intensity" in meta


def test_load_image_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image_12345.jpg")


def test_get_image_list_empty(tmp_path):
    imgs = get_image_list(tmp_path)
    assert isinstance(imgs, list)
    assert len(imgs) == 0


def test_extract_zip_files(tmp_path):
    import zipfile
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "sample content")
    
    count = extract_zip_files(tmp_path)
    assert count == 1
    assert (tmp_path / "test.txt").exists()


def test_load_experiment_image_synthetic():
    img, label = load_experiment_image(use_synthetic=True, height=64, width=64)
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64, 3)
    assert "synthetic" in str(label)


def test_download_kaggle_dataset_skips_when_populated(tmp_path):
    import cv2
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "dummy.png"), dummy_img)

    result_path = download_kaggle_dataset(target_dir=tmp_path, force_download=False)
    assert result_path == tmp_path
    imgs = get_image_list(tmp_path)
    assert len(imgs) == 1

