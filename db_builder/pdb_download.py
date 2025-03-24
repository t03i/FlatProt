# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Helper script to download PDB files
import os
import gzip
import httpx
from pathlib import Path
import logging
from typing import Dict, Any

from snakemake.script import snakemake
from httpx_retries import RetryTransport

URLS = [
    ("PDB", "https://files.rcsb.org/download/{pdb_id}.pdb.gz"),
    (
        "PDB",
        "https://ftp.ebi.ac.uk/pub/databases/rcsb/pdb-remediated/data/structures/divided/pdb/fm/pdb{pdb_id}.ent.gz",
    ),
]


def download_pdb_file(
    pdb_id: str,
    output_file: str,
    status_file: str,
    retry_count: int = 3,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Download a PDB file with retry logic using httpx and RetryTransport.

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
    output_path = Path(output_file)
    tmp_file = str(output_file) + ".gz"
    success = False
    error_message = ""

    # Ensure the output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Create a client with retry transport
    transport = RetryTransport(
        max_retries=retry_count,
        retry_backoff_factor=1.0,
    )

    client = httpx.Client(transport=transport, timeout=timeout)

    # Try RCSB PDB first, then PDBe as fallback
    for format, url in URLS:
        try:
            # Download compressed file
            response = client.get(url.format(pdb_id=pdb_id))
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

            client.close()
            success = True
            logging.info(f"Successfully downloaded PDB file {pdb_id}")
            break  # Success, exit the loop

        except (httpx.HTTPError, IOError, Exception) as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_message = (
                f"Error downloading PDB file {pdb_id} from {url}: {error_msg}"
            )
            logging.error(error_message)
            # Continue to the next URL if available

    # Close the client
    client.close()

    # If all URLs failed, create an empty PDB file to satisfy Snakemake
    if not success:
        with open(output_file, "w") as f:
            f.write(f"# Failed to download PDB {pdb_id}\n")
        logging.error(f"All download attempts failed for PDB {pdb_id}")

    # Write status file
    with open(status_file, "w") as f:
        if success:
            f.write("success")
        else:
            f.write(f"failed\n{error_message}")

    return {
        "success": success,
        "pdb_id": pdb_id,
        "error": error_message if not success else "",
    }


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
    output_file = snakemake.output.pdb_file
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
