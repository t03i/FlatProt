# %% [markdown]
# # FlatProt Runtime Measurement: Human Proteome
#
# **Goal:** This script measures the runtime of FlatProt's structural alignment and projection functionality on AlphaFold predicted structures from the human proteome.
#
# **Workflow:**
# 1.  Downloads AlphaFold PDB structures for UniProt IDs listed in an input file.
# 2.  Runs `flatprot align` (via internal API) using a FoldSeek database.
# 3.  Runs `flatprot project` (via internal API) if alignment is successful.
# 4.  Records alignment time, projection time, and any errors.
# 5.  Saves results to a TSV file.

# %% [markdown]
# ---
# ## Environment Setup for Google Colab
#
# The following cell checks if the notebook is running in Google Colab,
# downloads the shared setup script from GitHub, and runs it to install
# dependencies and download data.

# %%
import sys
import subprocess
from pathlib import Path
import os  # Keep os import if needed elsewhere

# Check if in Colab
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print("Running in Google Colab. Fetching and executing setup script...")
    # URL to the raw colab_setup.py script on GitHub (adjust branch if necessary)
    setup_script_url = (
        "https://raw.githubusercontent.com/t03i/FlatProt/main/examples/colab_setup.py"
    )
    setup_script_local_path = Path("colab_setup.py")

    # Download the setup script using wget
    print(f"Downloading {setup_script_url} to {setup_script_local_path}...")
    subprocess.run(
        ["wget", "-q", "-O", str(setup_script_local_path), setup_script_url],
        check=True,
    )
    print("Download complete.")

    # Ensure the current directory is in the Python path
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    # Import and run the setup function
    if setup_script_local_path.exists():
        # Import the downloaded script
        import colab_setup

        print("Running colab_setup.setup_colab_environment()...")
        colab_setup.setup_colab_environment()
        print("Colab setup script finished.")
        # Optional: Clean up the downloaded script to keep the env clean
        # setup_script_local_path.unlink(missing_ok=True)
    else:
        # This should not happen if wget was successful
        raise RuntimeError(
            f"Setup script {setup_script_local_path} not found after download attempt."
        )


# Define base_dir (works for both Colab and local)
# Assumes execution from the root of the repository or within examples/
base_dir = Path(".")

# --- Path Definitions ---
print(f"[INFO] Using base directory: {base_dir.resolve()}")
# Check if data exists after potential setup
data_dir_base = base_dir / "data"
if not data_dir_base.exists():
    print(
        f"[WARN] Data directory '{data_dir_base}' not found. Subsequent steps might fail.",
        file=sys.stderr,
    )
    # Optionally create it to prevent errors later if needed:
    # data_dir_base.mkdir(exist_ok=True)

tmp_dir_base = base_dir / "tmp"

# Ensure base tmp directory exists
tmp_dir_base.mkdir(parents=True, exist_ok=True)

# %%
# Import remaining libraries
import time
from typing import Optional, NamedTuple, Tuple
import shutil
import requests
import polars as pl
import tempfile

# Import FlatProt components (assuming installed by setup)
from flatprot.cli.align import align_structure_rotation, project_structure_svg
from flatprot.core.error import FlatProtError

print("[INFO] Imported remaining libraries and FlatProt components.")


class RuntimeResult(NamedTuple):
    """Results from a runtime measurement."""

    uniprot_id: str
    alignment_time: float
    projection_time: float
    error_type: Optional[str]
    has_match: bool


def download_alphafold_structure(uniprot_id: str, output_dir: Path) -> Optional[Path]:
    """Download AlphaFold structure for given UniProt ID.

    Args:
        uniprot_id: UniProt identifier
        output_dir: Directory to save PDB file

    Returns:
        Path to downloaded PDB file or None if download failed
    """
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    pdb_file = output_dir / f"{uniprot_id}_alphafold.pdb"

    try:
        response = requests.get(pdb_url, timeout=30)
        response.raise_for_status()

        pdb_file.write_text(response.text)
        return pdb_file
    except (requests.RequestException, IOError) as e:
        print(f"Failed to download structure for {uniprot_id}: {e}")
        return None


def measure_runtime(
    structure_file: Path,
    foldseek_db_path: Path,
    foldseek_executable: str = "foldseek",
    min_probability: float = 0.5,
) -> Tuple[float, float, bool, Optional[str]]:
    """Measure time for structural alignment and projection.

    Args:
        structure_file: Path to structure file
        foldseek_db_path: Path to FoldSeek database
        foldseek_executable: Path to FoldSeek executable
        min_probability: Minimum alignment probability threshold

    Returns:
        Tuple of (alignment time, projection time, whether match was found, error type if any)
    """
    error_type = None
    has_match = False
    alignment_time = 0.0
    projection_time = 0.0

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            matrix_path = tmp_path / "matrix.npy"
            info_path = tmp_path / "info.json"

            # Measure alignment time
            align_start = time.time()
            align_result = align_structure_rotation(
                structure_file=structure_file,
                matrix_out_path=matrix_path,
                info_out_path=info_path,
                foldseek_path=foldseek_executable,
                foldseek_db_path=foldseek_db_path,
                min_probability=min_probability,
            )
            alignment_time = time.time() - align_start

            if align_result == 0:
                has_match = True
                # Measure projection time
                proj_start = time.time()
                proj_result = project_structure_svg(
                    structure=structure_file,
                    matrix=matrix_path,
                    output=tmp_path / "output.svg",
                )
                projection_time = time.time() - proj_start

                if proj_result != 0:
                    error_type = "projection_error"
            else:
                error_type = "alignment_error"

    except FlatProtError as e:
        error_type = f"flatprot_error: {str(e)}"
    except Exception as e:
        error_type = f"unexpected_error: {str(e)}"

    return alignment_time, projection_time, has_match, error_type


def save_results(results: list[RuntimeResult], output_file: Path) -> None:
    """Save runtime results to TSV file.

    Args:
        results: List of runtime measurements
        output_file: Output file path
    """
    df = pl.DataFrame(
        {
            "uniprot_id": [r.uniprot_id for r in results],
            "alignment_time": [r.alignment_time for r in results],
            "projection_time": [r.projection_time for r in results],
            "total_time": [r.alignment_time + r.projection_time for r in results],
            "error_type": [r.error_type for r in results],
            "has_match": [r.has_match for r in results],
        }
    )

    if output_file.exists():
        existing_df = pl.read_csv(output_file, separator="\t")
        df = pl.concat([existing_df, df])

    df.write_csv(output_file, separator="\t")


def process_proteome(
    input_file: Path,
    output_file: Path,
    work_dir: Path,
    foldseek_db_path: Path,
    foldseek_executable: str = "foldseek",
    batch_size: int = 10,
) -> None:
    """Process proteome entries and measure alignment runtime.

    Args:
        input_file: TSV file with UniProt IDs
        output_file: Output file for results
        work_dir: Working directory for temporary files
        foldseek_db_path: Path to FoldSeek database
        foldseek_executable: Path to FoldSeek executable
        batch_size: Number of entries to process before saving
    """
    # Create working directories
    pdb_dir = work_dir / "structures"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Read input data
    df = pl.read_csv(input_file, separator="\t", has_header=True, comment_prefix="#")

    results = []
    processed_count = 0

    for row in df.iter_rows(named=True):
        uniprot_id = row["Entry"]

        # Skip if already processed
        if "alignment_time" in row and row["alignment_time"] is not None:
            continue

        # Download structure
        pdb_file = download_alphafold_structure(uniprot_id, pdb_dir)
        if not pdb_file:
            results.append(
                RuntimeResult(uniprot_id, -1.0, -1.0, "download_failed", False)
            )
            continue

        # Measure runtime
        alignment_time, projection_time, has_match, error = measure_runtime(
            pdb_file,
            foldseek_db_path,
            foldseek_executable,
        )

        results.append(
            RuntimeResult(uniprot_id, alignment_time, projection_time, error, has_match)
        )
        processed_count += 1

        # Save batch results
        if processed_count % batch_size == 0:
            save_results(results, output_file)
            results = []
            # Clean temporary files
            shutil.rmtree(pdb_dir)
            pdb_dir.mkdir(parents=True)

    # Save final results
    if results:
        save_results(results, output_file)

    # Final cleanup
    shutil.rmtree(pdb_dir)


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = Path("data/human_proteome.tsv")
    OUTPUT_FILE = Path("results/runtime_measurements.tsv")
    WORK_DIR = Path("work")
    FOLDSEEK_DB = Path("path/to/foldseek/db")
    FOLDSEEK_EXECUTABLE = "foldseek"  # or full path

    # Create directories
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Run measurements
    process_proteome(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        work_dir=WORK_DIR,
        foldseek_db_path=FOLDSEEK_DB,
        foldseek_executable=FOLDSEEK_EXECUTABLE,
    )
