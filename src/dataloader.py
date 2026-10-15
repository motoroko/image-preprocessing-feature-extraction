"""
Modul Data Loader dan Pengelolaan Gambar.

Menangani pemuatan citra dari path/dataset, konversi BGR-ke-RGB, 
pengecekan metadata citra, penanganan error, serta pembuatan citra demo sintetis.
"""

from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Union
import zipfile
import cv2
import numpy as np

from src.config import DATA_DIR, DATASET_SLUG, DEFAULT_SEED, SUPPORTED_EXTENSIONS


def get_image_list(
    dataset_dir: Union[str, Path] = DATA_DIR,
    extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS,
    recursive: bool = True
) -> List[Path]:
    """
    Mengambil daftar file citra dari direktori dataset yang memiliki ekstensi yang didukung.

    Parameters:
        dataset_dir: Path folder direktori dataset.
        extensions: Tuple ekstensi file yang valid (contoh: .jpg, .png).
        recursive: Apakah pencarian dilakukan secara rekursif ke subdirektori.

    Returns:
        List[Path]: Daftar path citra yang ditemukan, terurut alfabetis.
    """
    path = Path(dataset_dir)
    if not path.exists() or not path.is_dir():
        return []

    images = []
    pattern = "**/*" if recursive else "*"
    for p in path.glob(pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            images.append(p)

    return sorted(images)


def load_image(
    image_path: Union[str, Path],
    as_gray: bool = False
) -> np.ndarray:
    """
    Memuat satu gambar dari path dan mengembalikan representasi NumPy array.
    Konvensi internal:
    - Citra berwarna dikembalikan sebagai RGB (bukan BGR OpenCV).
    - Citra grayscale dikembalikan sebagai array 2D uint8.

    Parameters:
        image_path: Path file gambar.
        as_gray: Jika True, gambar dimuat dalam format grayscale (2D array).

    Returns:
        np.ndarray: Array gambar (H, W, 3) untuk RGB atau (H, W) untuk Grayscale.

    Raises:
        FileNotFoundError: Jika file gambar tidak ditemukan.
        ValueError: Jika file rusak atau tidak dapat dibaca oleh cv2.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"File gambar tidak ditemukan di path: {path.resolve()}\n"
            f"Pastikan path benar atau dataset sudah diletakkan di direktori data/."
        )

    # cv2.imread support with UTF-8 path handling on Windows
    try:
        # cv2.imdecode handles non-ascii paths properly on Windows
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(raw_bytes, flag)
    except Exception as e:
        raise ValueError(f"Gagal membaca file gambar {path}: {str(e)}")

    if img is None:
        raise ValueError(
            f"Gambar di {path} rusak atau format tidak didukung oleh cv2."
        )

    if not as_gray:
        # Konversi BGR OpenCV ke standar RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_image_metadata(image: np.ndarray) -> Dict[str, Union[str, int, float, Tuple]]:
    """
    Mengekstrak informasi metadata teknis dari array gambar.

    Parameters:
        image: Array NumPy citra (2D atau 3D).

    Returns:
        Dict berisi informasi:
            - shape: tuple dimensi
            - height: tinggi dalam piksel
            - width: lebar dalam piksel
            - channels: jumlah kanal warna (1 untuk gray, 3 untuk RGB)
            - dtype: tipe data array (misal uint8, float32)
            - min_intensity: nilai piksel minimum
            - max_intensity: nilai piksel maksimum
            - mean_intensity: rata-rata nilai intensitas
            - std_intensity: standar deviasi intensitas
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input harus berupa numpy.ndarray.")

    shape = image.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2] if image.ndim == 3 else 1

    return {
        "shape": shape,
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
    }


def select_random_image(
    dataset_dir: Union[str, Path] = DATA_DIR,
    seed: Optional[int] = DEFAULT_SEED,
    as_gray: bool = False
) -> Tuple[np.ndarray, Path]:
    """
    Memilih satu citra secara acak dari dataset dengan random seed yang dapat direproduksi.

    Parameters:
        dataset_dir: Direktori dataset.
        seed: Random seed untuk reproduktifitas.
        as_gray: Format grayscale jika True.

    Returns:
        Tuple[np.ndarray, Path]: Array citra dan path file yang terpilih.

    Raises:
        FileNotFoundError: Jika tidak ada citra yang ditemukan di dataset_dir.
    """
    images = get_image_list(dataset_dir)
    if not images:
        raise FileNotFoundError(
            f"Tidak ditemukan file citra yang didukung di '{dataset_dir}'.\n"
            f"Silakan unduh dataset atau gunakan create_synthetic_face() sebagai demo alternatif."
        )

    rng = np.random.default_rng(seed)
    selected_idx = rng.integers(0, len(images))
    selected_path = images[selected_idx]

    img = load_image(selected_path, as_gray=as_gray)
    return img, selected_path


def create_synthetic_face(
    height: int = 256,
    width: int = 256,
    seed: Optional[int] = DEFAULT_SEED
) -> np.ndarray:
    """
    Menghasilkan citra sintetis wajah sederhana (RGB) untuk pengujian dan demonstrasi 
    jika dataset Kaggle belum diunduh secara lokal.

    PENTING: Citra ini adalah data uji sintetis lokal, BUKAN bagian dari dataset Kaggle.

    Parameters:
        height: Tinggi citra dalam piksel.
        width: Lebar citra dalam piksel.
        seed: Seed jika ada variasi acak.

    Returns:
        np.ndarray: Citra RGB uint8 dengan pola wajah terstruktur.
    """
    # Background gradien lembut
    y, x = np.mgrid[0:height, 0:width]
    bg_r = np.clip(180 + (x / width) * 40, 0, 255).astype(np.uint8)
    bg_g = np.clip(190 + (y / height) * 30, 0, 255).astype(np.uint8)
    bg_b = np.clip(210 - (y / height) * 30, 0, 255).astype(np.uint8)
    img = np.stack([bg_r, bg_g, bg_b], axis=-1)

    # Pusat wajah
    cx, cy = width // 2, height // 2

    # Kepala (oval kulit)
    head_color = (240, 200, 160)
    cv2.ellipse(img, (cx, cy), (int(width * 0.35), int(height * 0.42)), 0, 0, 360, head_color, -1)
    cv2.ellipse(img, (cx, cy), (int(width * 0.35), int(height * 0.42)), 0, 0, 360, (180, 140, 100), 2)

    # Mata kiri & kanan
    eye_y = int(cy - height * 0.1)
    eye_offset_x = int(width * 0.15)
    eye_color = (255, 255, 255)
    pupil_color = (40, 70, 120)

    for ex in [cx - eye_offset_x, cx + eye_offset_x]:
        cv2.ellipse(img, (ex, eye_y), (int(width * 0.07), int(height * 0.04)), 0, 0, 360, eye_color, -1)
        cv2.circle(img, (ex, eye_y), int(width * 0.025), pupil_color, -1)
        cv2.circle(img, (ex + 2, eye_y - 2), int(width * 0.008), (255, 255, 255), -1)  # highlight

    # Alis
    brow_color = (80, 50, 30)
    cv2.line(img, (cx - eye_offset_x - 15, eye_y - 18), (cx - eye_offset_x + 15, eye_y - 14), brow_color, 3)
    cv2.line(img, (cx + eye_offset_x - 15, eye_y - 14), (cx + eye_offset_x + 15, eye_y - 18), brow_color, 3)

    # Hidung
    nose_color = (210, 160, 120)
    cv2.line(img, (cx, eye_y + 10), (cx - 5, cy + 15), nose_color, 2)
    cv2.line(img, (cx - 5, cy + 15), (cx + 8, cy + 15), nose_color, 2)

    # Mulut (tersenyum)
    mouth_y = int(cy + height * 0.2)
    mouth_color = (190, 60, 70)
    cv2.ellipse(img, (cx, mouth_y), (int(width * 0.14), int(height * 0.06)), 0, 0, 180, mouth_color, 3)

    # Pipi blush
    blush_color = (245, 175, 165)
    cv2.ellipse(img, (cx - eye_offset_x, cy + 15), (int(width * 0.07), int(height * 0.03)), 0, 0, 360, blush_color, -1)
    cv2.ellipse(img, (cx + eye_offset_x, cy + 15), (int(width * 0.07), int(height * 0.03)), 0, 0, 360, blush_color, -1)

    return img


def extract_zip_files(target_dir: Union[str, Path]) -> int:
    """
    Mengekstrak seluruh file .zip yang ditemukan di target_dir (secara rekursif).

    Parameters:
        target_dir: Direktori tempat pencarian file .zip.

    Returns:
        int: Jumlah file zip yang berhasil diekstrak.
    """
    target_path = Path(target_dir)
    if not target_path.exists():
        return 0

    extracted_count = 0
    for zip_file in list(target_path.rglob("*.zip")):
        try:
            print(f"Mengekstrak file zip: {zip_file.name} ke {target_path}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_path)
            extracted_count += 1
        except Exception as e:
            print(f"Peringatan: Gagal mengekstrak {zip_file.name}: {e}")

    return extracted_count


def download_kaggle_dataset(
    target_dir: Union[str, Path] = DATA_DIR,
    dataset_slug: str = DATASET_SLUG,
    force_download: bool = False
) -> Path:
    """
    Mengunduh dataset dari Kaggle dan mengekstraknya ke target_dir jika direktori masih kosong.

    Alur:
    1. Mengecek apakah target_dir sudah memuat citra. Jika ya dan force_download=False, proses unduh dilewati.
    2. Mengecek apakah terdapat file archive .zip di target_dir dan mengekstraknya jika ada.
    3. Jika direktori masih belum memuat citra, mengunduh dataset dari Kaggle via pustaka `kagglehub`
       (mendukung dataset publik tanpa kewajiban autentikasi lokal), lalu menyalin isinya ke target_dir.
    4. Mendukung fallback ke Kaggle CLI jika kagglehub tidak tersedia.
    5. Mengekstrak kembali jika terdapat file zip yang baru diunduh.

    Parameters:
        target_dir: Direktori target untuk menyimpan citra dataset (default: DATA_DIR).
        dataset_slug: Slug dataset Kaggle (format: 'owner/dataset-name').
        force_download: Jika True, tetap mengunduh ulang meskipun citra sudah ada.

    Returns:
        Path: Path direktori dataset.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 1. Ekstrak zip lokal jika sudah ada di folder target
    extract_zip_files(target_path)

    # Cek apakah citra sudah tersedia
    existing_images = get_image_list(target_path)
    if existing_images and not force_download:
        print(f"Dataset sudah tersedia di '{target_path}' ({len(existing_images)} citra ditemukan). Pengunduhan dilewati.")
        return target_path

    print(f"Mengunduh dataset Kaggle '{dataset_slug}' ke '{target_path}'...")
    download_success = False

    # 2. Coba unduh menggunakan kagglehub
    try:
        import kagglehub
        cache_path = Path(kagglehub.dataset_download(dataset_slug))
        print(f"Dataset berhasil diunduh via kagglehub ke cache: {cache_path}")

        # Salin isi dari cache_path ke target_path
        if cache_path.is_dir() and cache_path.resolve() != target_path.resolve():
            for item in cache_path.iterdir():
                dest = target_path / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            print(f"File dataset berhasil disalin ke direktori: {target_path}")

        download_success = True
    except Exception as e:
        print(f"Pemberitahuan: Unduhan via kagglehub menemui kendala ({e}). Mencoba fallback...")

    # 3. Fallback ke Kaggle CLI jika kagglehub gagal
    if not download_success:
        import subprocess
        try:
            cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_path), "--unzip"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Berhasil mengunduh via Kaggle CLI:", result.stdout.strip())
            download_success = True
        except Exception as e:
            print(f"Pemberitahuan: Unduhan via Kaggle CLI tidak tersedia atau gagal ({e}).")

    # 4. Ekstrak zip yang mungkin baru saja terunduh
    extract_zip_files(target_path)

    # 5. Verifikasi citra yang ditemukan
    final_images = get_image_list(target_path)
    if final_images:
        print(f"Dataset siap! Berhasil memuat {len(final_images)} citra di '{target_path}'.")
    else:
        print(
            f"Perhatian: Tidak ditemukan file citra yang didukung di '{target_path}'.\n"
            f"Pastikan koneksi internet aktif atau letakkan file gambar/zip langsung di folder data/."
        )

    return target_path


def load_experiment_image(
    dataset_dir: Union[str, Path] = DATA_DIR,
    use_synthetic: bool = False,
    seed: Optional[int] = DEFAULT_SEED,
    height: int = 256,
    width: int = 256,
    auto_download: bool = True
) -> Tuple[np.ndarray, Union[Path, str]]:
    """
    Memuat citra untuk eksperimen dengan skema pemilihan: citra sintetis vs dataset Kaggle.

    Parameters:
        dataset_dir: Direktori dataset lokal.
        use_synthetic: Jika True, gunakan create_synthetic_face().
                       Jika False, gunakan citra nyata dari dataset Kaggle.
        seed: Random seed untuk reproduktifitas.
        height: Tinggi citra jika menggunakan data sintetis.
        width: Lebar citra jika menggunakan data sintetis.
        auto_download: Jika True dan dataset_dir kosong saat use_synthetic=False,
                       otomatis mengunduh dari Kaggle dan mengekstraknya.

    Returns:
        Tuple[np.ndarray, Union[Path, str]]: Array gambar (RGB) dan path / penanda nama gambar.
    """
    if use_synthetic:
        print("Mode: Menggunakan citra sintetis wajah (create_synthetic_face)...")
        img = create_synthetic_face(height=height, width=width, seed=seed)
        return img, "synthetic_face_256x256"

    images = get_image_list(dataset_dir)
    if not images and auto_download:
        print(f"Direktori '{dataset_dir}' masih kosong. Mengunduh dataset dari Kaggle...")
        download_kaggle_dataset(target_dir=dataset_dir)
        images = get_image_list(dataset_dir)

    if images:
        return select_random_image(dataset_dir=dataset_dir, seed=seed)
    else:
        print("Peringatan: Tidak ada citra di dataset lokal. Menggunakan citra sintetis sebagai fallback.")
        img = create_synthetic_face(height=height, width=width, seed=seed)
        return img, "synthetic_face_fallback"

