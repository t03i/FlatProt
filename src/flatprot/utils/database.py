# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Database management utilities for FlatProt alignment features."""

import tempfile
import shutil
from pathlib import Path
from typing import Optional
import httpx
import asyncio

from ..core.logger import logger

# Default database settings
DEFAULT_DB_URL = "https://flatprot.org/databases/alignment_db.tar.gz"
DEFAULT_DB_DIR = Path.home() / ".flatprot" / "databases" / "alignment"

# Files that must be present in a valid alignment database
REQUIRED_DB_FILES = ["alignments.h5", "database_info.json", "foldseek/db.index"]


def ensure_database_available(
    database_path: Optional[Path] = None, force_download: bool = False
) -> Path:
    """Ensure the alignment database is available, downloading if necessary.

    Args:
        database_path: Optional custom path to the database. If not provided,
                      the default location will be used.
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
        # Run the async download function
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
        output_dir: Directory where the database should be downloaded.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    logger.info(f"Downloading alignment database from {DEFAULT_DB_URL}")

    # Create a temporary directory for downloading
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "alignment_db.tar.gz"

        try:
            # Download the archive using httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(DEFAULT_DB_URL)
                response.raise_for_status()

                # Save the downloaded content
                with open(archive_path, "wb") as f:
                    f.write(response.content)

            # Extract the archive to the output directory
            logger.info(f"Extracting alignment database to {output_dir}")
            shutil.unpack_archive(archive_path, output_dir)

            # Get extracted files/folders (for validation)
            extracted_items = list(output_dir.glob("*"))
            if not extracted_items:
                raise RuntimeError("No files were extracted from the database archive")

            logger.debug(
                f"Extracted {len(extracted_items)} items into database directory"
            )

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
