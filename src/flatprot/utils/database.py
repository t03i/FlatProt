# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Database management utilities for FlatProt alignment features."""

import tempfile
import shutil
from pathlib import Path
from typing import Optional
import httpx
import asyncio
import platformdirs

from ..core.logger import logger

# Default database settings
DEFAULT_DB_URL = "https://zenodo.org/records/15264810/files/alignment_db.zip?download=1"
APP_NAME = "flatprot"
APP_AUTHOR = "rostlab"
DEFAULT_DB_DIR = (
    Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR)) / "databases" / "alignment"
)

# Files that must be present in a valid alignment database
REQUIRED_DB_FILES = ["alignments.h5", "database_info.json", "foldseek/db.index"]


def ensure_database_available(
    database_path: Optional[Path] = None, force_download: bool = False
) -> Path:
    """Ensure the alignment database is available, downloading if necessary.

    Args:
        database_path: Optional custom path to the database. If not provided,
                      the default location (`{DEFAULT_DB_DIR}`) will be used.
        force_download: If True, force re-download even if database exists.

    Returns:
        Path to the available database directory.

    Raises:
        RuntimeError: If the database could not be obtained or is invalid.
    """
    # Determine the database path
    db_path = database_path or DEFAULT_DB_DIR

    # If database exists and is valid, return it unless force_download is True
    if db_path.exists() and not force_download:
        if validate_database(db_path):
            logger.info(f"Using existing alignment database at {db_path}")
            return db_path
        else:
            logger.warning(
                f"Existing database at {db_path} is [bold]invalid[/bold], will download"
            )

    # Create directory if it doesn't exist
    db_path.mkdir(parents=True, exist_ok=True)

    # Download the database
    try:
        # Run the async download function, handling existing event loops
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Existing asyncio loop detected, scheduling download.")
            loop.run_until_complete(download_database(db_path))
        except RuntimeError:  # No event loop running
            logger.debug("No asyncio loop running, using asyncio.run for download.")
            asyncio.run(download_database(db_path))

        if validate_database(db_path):
            logger.info(
                f"[bold]Successfully downloaded[/bold] alignment database to {db_path}"
            )
            return db_path
        else:
            raise RuntimeError("Downloaded database is invalid")
    except Exception as e:
        logger.error(f"Failed to download alignment database: {str(e)}")
        raise RuntimeError(f"Could not obtain alignment database: {str(e)}")


async def download_database(output_dir: Path) -> None:
    """Download and extract the alignment database from the remote URL.

    Args:
        output_dir: Directory where the database should be downloaded and extracted.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    logger.info(f"Downloading alignment database from {DEFAULT_DB_URL}")

    # Create a temporary directory for downloading and extraction
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "alignment_db.zip"
        extract_temp_path = tmp_path / "extracted"  # Extract to a sub-folder
        extract_temp_path.mkdir()

        try:
            # Download the archive using httpx, disable SSL verification
            logger.warning("Disabling SSL verification for database download.")
            async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
                response = await client.get(DEFAULT_DB_URL)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    f.write(response.content)

            # Extract the archive to the temporary extraction path
            logger.info(
                f"Extracting archive to temporary location: {extract_temp_path}"
            )
            shutil.unpack_archive(archive_path, extract_temp_path)

            # Define the path to the expected data directory within the extraction
            source_data_dir = extract_temp_path / "alignment_db"

            # Check if the expected data directory exists
            if not source_data_dir.is_dir():
                raise RuntimeError(
                    f"Required 'alignment_db' directory not found in extracted archive at {extract_temp_path}"
                )

            # Move contents from the source data directory to the final output directory
            logger.debug(f"Moving contents from {source_data_dir} to {output_dir}")
            item_count = 0
            for item in source_data_dir.iterdir():
                target_path = output_dir / item.name
                try:
                    shutil.move(str(item.resolve()), str(target_path.resolve()))
                    item_count += 1
                except Exception as move_err:
                    logger.error(
                        f"Error moving {item.name} to {target_path}: {move_err}"
                    )
                    raise RuntimeError(
                        f"Failed to move extracted file/directory: {move_err}"
                    ) from move_err

            if item_count == 0:
                logger.warning(
                    f"The extracted source directory {source_data_dir} was empty."
                )

            logger.debug(f"Successfully moved {item_count} items to {output_dir}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during database download: {str(e)}")
            raise RuntimeError(f"Database download failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to download or extract database: {str(e)}")
            raise RuntimeError(f"Database download failed: {str(e)}")


def validate_database(db_path: Path) -> bool:
    """Check if the alignment database at the given path is valid.

    A valid database must contain all the required files.

    Args:
        db_path: Path to the alignment database directory.

    Returns:
        True if the database is valid, False otherwise.
    """
    logger.debug(f"Validating alignment database at {db_path}")

    # Check if the directory exists
    if not db_path.exists() or not db_path.is_dir():
        logger.warning(f"Database directory {db_path} does not exist")
        return False

    # Check for required files
    missing_files = []
    for required_file in REQUIRED_DB_FILES:
        file_path = db_path / required_file
        if not file_path.exists():
            missing_files.append(required_file)

    if missing_files:
        logger.warning(
            f"Database at {db_path} is missing required files: [bold]{', '.join(missing_files)}[/bold]"
        )
        return False

    logger.debug(f"Database at {db_path} is [bold]valid[/bold]")
    return True
