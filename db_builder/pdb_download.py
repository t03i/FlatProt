# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Helper script to download PDB files
import os
import gzip
import httpx
from pathlib import Path
import logging
from typing import Dict, Any
import json

from snakemake.script import snakemake
from httpx_retries import RetryTransport, Retry

URLS = [
    # Try CIF format first
    ("CIF", "https://files.rcsb.org/download/{pdb_id}.cif.gz"),
    (
        "CIF",
        "https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/mmCIF/{middle_chars}/{pdb_id}.cif.gz",
    ),
    # Then try PDB format as fallback
    ("PDB", "https://files.rcsb.org/download/{pdb_id}.pdb.gz"),
    (
        "PDB",
        "https://ftp.ebi.ac.uk/pub/databases/rcsb/pdb-remediated/data/structures/divided/pdb/{middle_chars}/pdb{pdb_id}.ent.gz",
    ),
]


def get_middle_chars(pdb_id: str) -> str:
    """Extract the middle characters of a PDB ID for directory structure in some repositories.

    Args:
        pdb_id: The 4-character PDB ID

    Returns:
        The 2 middle characters of the PDB ID
    """
    if len(pdb_id) >= 4:
        return pdb_id[1:3].lower()
    return ""


def download_pdb_file(
    pdb_id: str,
    output_file: str,
    status_file: str,
    retry_count: int = 3,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Download a structure file (CIF or PDB) with retry logic using httpx and RetryTransport.

    Args:
        pdb_id: The PDB ID to download
        output_file: Path to save the file
        status_file: Path to save the status (success/failure)
        retry_count: Number of retry attempts for failed downloads
        timeout: Timeout for HTTP requests in seconds

    Returns:
        Dict with status information for Snakemake reporting
    """
    pdb_id = pdb_id.lower()
    middle_chars = get_middle_chars(pdb_id)
    output_path = Path(output_file)
    tmp_file = str(output_file) + ".gz"
    success = False
    error_message = ""
    downloaded_format = None

    # Ensure the output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Create a retry strategy
    retry = Retry(total=retry_count, backoff_factor=1.0)

    # Try each URL in order (CIF first, then PDB)
    for format, url in URLS:
        try:
            # Format URL with PDB ID and middle characters
            formatted_url = url.format(pdb_id=pdb_id, middle_chars=middle_chars)

            # Create a new client for each request with the retry transport
            transport = RetryTransport(retry=retry)
            with httpx.Client(transport=transport, timeout=timeout) as client:
                # Download compressed file
                response = client.get(formatted_url)
                response.raise_for_status()

                # Save compressed data to file
                with open(tmp_file, "wb") as f:
                    f.write(response.content)

            # Decompress
            with gzip.open(tmp_file, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    f_out.write(f_in.read())

            # Clean up
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)

            downloaded_format = format
            success = True
            logging.info(f"Successfully downloaded {format} file for {pdb_id}")
            break  # Success, exit the loop

        except (httpx.HTTPError, IOError, Exception) as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_message = (
                f"Error downloading {format} file {pdb_id} from {url}: {error_msg}"
            )
            logging.error(error_message)
            # Continue to the next URL if available

    # If all URLs failed, create an empty file to satisfy Snakemake
    if not success:
        with open(output_file, "w") as f:
            f.write(f"# Failed to download structure for {pdb_id}\n")
        logging.error(f"All download attempts failed for structure {pdb_id}")

    # Write status file with format information as JSON
    status_data = {
        "success": success,
        "pdb_id": pdb_id,
        "format": downloaded_format if success else None,
        "error": error_message if not success else "",
    }

    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

    return status_data


def main() -> None:
    """
    Main entry point for the Snakemake script.

    This function handles downloading a single PDB file as specified by Snakemake wildcards.
    It reads parameters from the Snakemake object and calls the download function.

    Returns:
        None
    """
    # Get parameters from Snakemake
    pdb_id = snakemake.params.pdb_id
    output_file = snakemake.output.struct_file
    status_file = snakemake.output.status

    # Get configuration parameters
    retry_count = snakemake.params.get("retry_count", 3)
    timeout = snakemake.params.get("timeout", 30)

    # Download the PDB file
    download_pdb_file(
        pdb_id=pdb_id,
        output_file=output_file,
        status_file=status_file,
        retry_count=retry_count,
        timeout=timeout,
    )


# Run with snakemake object
if __name__ == "__main__":
    main()
