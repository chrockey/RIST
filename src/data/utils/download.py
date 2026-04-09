"""Download utilities for datasets from Google Drive."""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Google Drive folder ID for KeypointNet dataset
KEYPOINTNET_FOLDER_ID = "1_d1TzZEF25Wy5kRj5ZugrgGeyf7xxu8F"


def check_gdown_installed() -> bool:
    """Check if gdown is installed."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def download_folder_from_gdrive(
    folder_id: str,
    output_dir: Path,
    quiet: bool = False,
    skip_dirs: list[str] | None = None,
) -> Path:
    """Download a folder from Google Drive.

    Args:
        folder_id: Google Drive folder ID
        output_dir: Directory to save files
        quiet: Suppress download progress
        skip_dirs: List of directory names to skip downloading (for already extracted zips)

    Returns:
        Path to downloaded directory
    """
    import gdown

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = skip_dirs or []

    # Get list of files to download
    files_to_download = gdown.download_folder(url, output=str(output_dir), quiet=True, skip_download=True)

    if files_to_download is None:
        raise RuntimeError(f"Failed to get file list from Google Drive folder: {folder_id}")

    # Filter out zip files whose extracted directories already exist
    for file_info in files_to_download:
        filename = Path(file_info.path).name
        # Skip zip files if their extracted directory exists
        if filename.endswith(".zip"):
            extracted_name = filename[:-4]  # Remove .zip extension
            if extracted_name in skip_dirs:
                logger.info(f"Skipping {filename} (already extracted)")
                print(f"Skipping {filename} (already extracted)")
                continue

        # Download the file
        local_path = output_dir / file_info.path
        if local_path.exists():
            logger.info(f"Skipping {filename} (already exists)")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=file_info.id, output=str(local_path), quiet=quiet, resume=True)

    return output_dir


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file.

    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extraction complete")


def download_keypointnet(
    output_dir: Path,
    category: Optional[str] = None,
    category_id: Optional[str] = None,
    quiet: bool = False,
    force: bool = False,
) -> Path:
    """Download KeypointNet dataset from Google Drive.

    Args:
        output_dir: Directory to save the dataset
        category: Category name (e.g., 'airplane'). Currently unused, downloads full dataset.
        category_id: Category ID (e.g., '02691156'). Currently unused, downloads full dataset.
        quiet: Suppress download progress
        force: If True, delete existing data and re-download

    Returns:
        Path to dataset directory
    """
    if not check_gdown_installed():
        raise ImportError(
            "gdown is required for auto-download. "
            "Install it with: pip install gdown"
        )

    output_dir = Path(output_dir)

    if force and output_dir.exists():
        print(f"Removing existing dataset at {output_dir}...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading KeypointNet dataset to {output_dir}")
    print(f"Downloading KeypointNet dataset to {output_dir}...")
    print("This may take several minutes depending on your connection.")

    # Check which zip files are already extracted
    skip_dirs = []
    zip_to_dir = {
        "pcds.zip": "pcds",
        "ShapeNetCore.v2.ply.zip": "ShapeNetCore.v2.ply",
    }
    for zip_name, dir_name in zip_to_dir.items():
        if (output_dir / dir_name).exists():
            skip_dirs.append(dir_name)

    # Download from Google Drive folder
    download_folder_from_gdrive(
        KEYPOINTNET_FOLDER_ID,
        output_dir,
        quiet=quiet,
        skip_dirs=skip_dirs,
    )

    # Handle any zip files that need extraction
    for zip_file in output_dir.glob("*.zip"):
        extracted_dir = output_dir / zip_file.stem
        if not extracted_dir.exists():
            print(f"Extracting {zip_file.name}... (this may take a while)")
            extract_zip(zip_file, zip_file.parent)
            zip_file.unlink()
            print(f"Extraction complete: {zip_file.name}")

    logger.info(f"KeypointNet dataset downloaded successfully to {output_dir}")
    print("Download complete!")
    return output_dir


def _get_rank() -> int:
    """Get current process rank for DDP."""
    import os
    # Check common DDP environment variables
    for var in ["RANK", "LOCAL_RANK", "SLURM_PROCID"]:
        if var in os.environ:
            return int(os.environ[var])
    return 0


def download_splits(data_dir: Path, category_id: str) -> None:
    """Download split files from KeypointNet GitHub repository.

    Downloads splits for ALL KeypointNet categories at once, so subsequent
    category changes don't require re-downloading.

    Args:
        data_dir: Root directory of KeypointNet dataset
        category_id: Category ID (e.g., '02691156') - used for DDP sync
    """
    import time
    import urllib.request

    from src.utils.common import KPN_CATEGORIES, NAMES2ID

    rank = _get_rank()
    splits_dir = data_dir / "splits" / category_id

    # Only rank 0 downloads
    if rank == 0:
        base_url = "https://raw.githubusercontent.com/qq456cvb/KeypointNet/master/splits"

        for split in ["train", "val", "test"]:
            # Check if all categories already have this split
            all_exist = all(
                (data_dir / "splits" / NAMES2ID[cat] / f"{split}.txt").exists()
                for cat in KPN_CATEGORIES
            )
            if all_exist:
                continue

            # Download full split file once
            url = f"{base_url}/{split}.txt"
            logger.info(f"Downloading {split} split from {url}")
            print(f"Downloading {split} split for all categories...")

            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read().decode("utf-8")

                # Process all KeypointNet categories
                for cat_name in KPN_CATEGORIES:
                    cat_id = NAMES2ID[cat_name]
                    cat_splits_dir = data_dir / "splits" / cat_id
                    split_file = cat_splits_dir / f"{split}.txt"

                    if split_file.exists():
                        continue

                    cat_splits_dir.mkdir(parents=True, exist_ok=True)

                    # Filter lines that match category_id (format: "category_id-model_id")
                    filtered_lines = []
                    for line in content.strip().split("\n"):
                        line = line.strip()
                        if line.startswith(f"{cat_id}-"):
                            # Extract model_id (remove category_id prefix)
                            model_id = line[len(cat_id) + 1:]
                            filtered_lines.append(model_id)

                    with open(split_file, "w") as f:
                        f.write("\n".join(filtered_lines))

                    logger.info(f"Saved {len(filtered_lines)} entries to {split_file}")
                    print(f"  {cat_name} {split}: {len(filtered_lines)} samples")

            except Exception as e:
                logger.warning(f"Failed to download {split} split: {e}")
                print(f"Warning: Failed to download {split} split: {e}")

    # Other ranks wait for files to appear
    else:
        for split in ["train", "val", "test"]:
            split_file = splits_dir / f"{split}.txt"
            while not split_file.exists():
                time.sleep(0.1)
