# %% [markdown]
# # FlatProt Runtime Benchmark: Human Proteome (Refactored)
#
# **Goal:** This notebook benchmarks FlatProt's structural projection performance on the human proteome using AlphaFold predicted structures.
#
# **Benchmarks:**
# 1. **Inertia Projection**: Time to read, transform (inertia matrix), project orthographically, create scene, and render SVG.
# 2. **Alignment + Projection**: Time to read, align against default DB (Foldseek), and project with the alignment rotation matrix.
#
# **Workflow:**
# 1. **Setup & Configuration**: Set up environment, paths, and parameters.
# 2. **Data Loading & Sampling**: Load human proteome data and sample UniProt IDs.
# 3. **Data Preparation**: Download AlphaFold structures and enrich with DSSP.
# 4. **Benchmarking**: Run inertia and alignment benchmarks.
# 5. **Results & Plotting**: Save and visualize benchmark results.
#
# ---

# %% [markdown]
# ## 1. Environment Setup & Initial Imports
#
# Configure logging, define base paths, and import essential libraries for setup and data handling.

# %%
import sys
import subprocess
from pathlib import Path
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Check if in Colab and run setup if necessary
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    log.info("Running in Google Colab. Setting up environment...")
    # URL to the raw colab_setup.py script on GitHub
    setup_script_url = "https://raw.githubusercontent.com/rostlab/FlatProt/main/examples/colab_setup.py"
    setup_script_local_path = Path("colab_setup.py")

    # Download the setup script
    if not setup_script_local_path.exists():
        log.info(f"Downloading {setup_script_url} to {setup_script_local_path}...")
        try:
            subprocess.run(
                ["wget", "-q", "-O", str(setup_script_local_path), setup_script_url],
                check=True,
            )
            log.info("Download complete.")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to download setup script: {e}")
            raise

    # Ensure the current directory is in the Python path
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    # Import and run the setup function
    if setup_script_local_path.exists():
        try:
            import colab_setup

            log.info("Running colab_setup.setup_colab_environment()...")
            colab_setup.setup_colab_environment()
            log.info("Colab setup complete.")
        except ImportError:
            log.error("Failed to import colab_setup after downloading.")
            raise
        except Exception as e:
            log.error(f"Error during Colab setup: {e}")
            raise
    else:
        raise RuntimeError(
            f"Setup script {setup_script_local_path} not found after download attempt."
        )

# Define base directory (works for both Colab and local)
# Determine base directory assuming script is in 'examples' or repo root
if Path.cwd().name == "examples":
    base_dir = Path.cwd().parent
else:
    base_dir = Path.cwd()  # Assume repo root if not in examples

# Set up directories relative to base_dir
data_dir = base_dir / "data"
results_dir = base_dir / "tmp" / "runtime" / "results"
tmp_dir = base_dir / "tmp" / "runtime"
# Separate raw and enriched structure directories for clarity
raw_structures_dir = tmp_dir / "structures_raw"
enriched_structures_dir = tmp_dir / "structures_dssp"

# Create necessary directories
for directory in [
    data_dir,
    results_dir,
    tmp_dir,
    raw_structures_dir,
    enriched_structures_dir,
]:
    directory.mkdir(parents=True, exist_ok=True)

log.info(f"Using base directory: {base_dir.resolve()}")
log.info(f"Data directory: {data_dir.resolve()}")
log.info(f"Results directory: {results_dir.resolve()}")
log.info(f"Temporary directory: {tmp_dir.resolve()}")
log.info(f"Raw structures directory: {raw_structures_dir.resolve()}")
log.info(f"Enriched structures directory: {enriched_structures_dir.resolve()}")

# %% [markdown]
# ## 2. Configuration Parameters
#
# Define parameters for data sampling, benchmarking, and file paths.

# %%
# --- General Config ---
HUMAN_PROTEOME_FILE = data_dir / "human_proteome.tsv"
RESULTS_FILE = results_dir / "runtime_benchmark_results.tsv"  # Changed filename
SAMPLED_IDS_FILE = results_dir / "sampled_uniprot_ids.txt"  # Changed filename
TARGET_SAMPLE_SIZE = 1000  # Number of proteins to aim for in the final benchmark
RANDOM_SEED = 42  # For reproducible sampling
NUM_BINS = 10  # For stratified sampling by length
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 1000

# --- Parallelism & Batching ---
# Use system's CPU count or default to 4 for parallel tasks
# Set lower if resource constraints exist (e.g., memory)
NUM_WORKERS = os.cpu_count() or 4
BENCHMARK_BATCH_SIZE = 50  # How often to save results during benchmarking

# --- Alignment Specific Config ---
FOLDSEEK_EXECUTABLE = "foldseek"  # Assumes foldseek is in PATH
# Set path to the PARENT directory of the Foldseek DB
# Leave as None to use default location (~/.cache/flatprot/db) and download if needed
FOLDSEEK_DB_PATH = None
MIN_ALIGN_PROBABILITY = (
    0.5  # Minimum alignment probability threshold for 'family' method
)

log.info(f"Target sample size: {TARGET_SAMPLE_SIZE}")
log.info(f"Number of workers for parallel tasks: {NUM_WORKERS}")
log.info(f"Results will be saved to: {RESULTS_FILE}")
log.info(f"Sampled IDs list will be saved to: {SAMPLED_IDS_FILE}")

# %% [markdown]
# ## 3. Load Human Proteome Data & Perform Sampling
#
# Load the UniProt IDs and lengths from the TSV file. Perform stratified sampling based on protein length to select a representative subset for benchmarking.

# %%
import polars as pl
from typing import Optional, List

log.info("--- Starting Data Loading and Sampling ---")

# Initialize variables
proteome_df_filtered: Optional[pl.DataFrame] = None
uniprot_ids_to_process: List[str] = []
id_column_name: Optional[str] = None
length_column_name: str = "Length"  # Default expected column name

if not HUMAN_PROTEOME_FILE.exists():
    log.error(f"Input file not found: {HUMAN_PROTEOME_FILE}")
    log.error(
        "Please place the UniProt TSV file for the human proteome (including 'Length' column) "
        f"in the '{data_dir.name}' directory ({data_dir})."
    )
    log.info(
        "Example command to download reviewed human proteins with accession and length:"
    )
    log.info(
        f"wget -O \"{HUMAN_PROTEOME_FILE}\" 'https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=%28%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29%29&fields=accession%2Clength'"
    )
    # Consider raising an error or exiting if the file is critical and not found
    # raise FileNotFoundError(f"Input file not found: {HUMAN_PROTEOME_FILE}")
else:
    # Removed the outer try...except block, allowing errors during loading/sampling to propagate
    log.info(f"Loading UniProt IDs and lengths from {HUMAN_PROTEOME_FILE}...")
    proteome_df = pl.read_csv(
        HUMAN_PROTEOME_FILE,
        separator="\t",
        has_header=True,
        comment_prefix="#",
        schema_overrides={"Length": pl.Int64},
    )
    log.info(f"Initial rows loaded: {proteome_df.height}")

    id_column_name = "Entry"

    # Filter out entries with missing lengths or IDs and ensure uniqueness
    proteome_df_filtered = proteome_df.filter(
        (
            pl.col(length_column_name) > 3  # Ensure length is positive
        )
    ).unique(subset=[id_column_name], keep="first")  # Keep only unique IDs

    total_unique_proteins = proteome_df_filtered.height
    log.info(
        f"Loaded {total_unique_proteins} unique valid proteins with positive lengths."
    )

    if total_unique_proteins == 0:
        log.warning(
            "No valid protein entries with IDs and positive lengths found. Cannot proceed with sampling."
        )
        # Reset dataframe to None to indicate failure
        proteome_df_filtered = None
    elif total_unique_proteins < TARGET_SAMPLE_SIZE:
        log.warning(
            f"Total unique valid proteins ({total_unique_proteins}) is less than target sample size ({TARGET_SAMPLE_SIZE}). Using all available proteins for initial processing list."
        )
        uniprot_ids_to_process = proteome_df_filtered.get_column(
            id_column_name
        ).to_list()
        # Add length_bin column even when not sampling for consistency
        proteome_df_filtered = proteome_df_filtered.with_columns(
            pl.lit("all").alias("length_bin")  # Assign a single bin label
        )
    else:
        log.info(
            f"Performing stratified sampling to select {TARGET_SAMPLE_SIZE} proteins for initial processing list..."
        )
        # Create length bins (e.g., deciles)
        try:
            proteome_df_filtered = proteome_df_filtered.with_columns(
                pl.col(length_column_name)
                .qcut(
                    NUM_BINS,
                    labels=[f"bin_{i + 1}" for i in range(NUM_BINS)],
                    allow_duplicates=True,
                )
                .alias("length_bin")
            )
        except Exception as e:
            log.error(
                f"Error creating quantile bins (qcut): {e}. This might happen with highly skewed length distributions or few data points. Falling back to using all proteins."
            )
            uniprot_ids_to_process = proteome_df_filtered.get_column(
                id_column_name
            ).to_list()
            proteome_df_filtered = proteome_df_filtered.with_columns(
                pl.lit("all").alias("length_bin")  # Assign a single bin label
            )

        # If binning succeeded, proceed with sampling
        if "length_bin" in proteome_df_filtered.columns and not uniprot_ids_to_process:
            # Calculate samples per bin proportionally, handling rounding carefully
            bin_counts = proteome_df_filtered.group_by(
                "length_bin", maintain_order=True
            ).len()
            bin_counts = bin_counts.with_columns(
                (pl.col("len") / pl.sum("len") * TARGET_SAMPLE_SIZE).alias(
                    "target_float"
                )
            )
            bin_counts = bin_counts.with_columns(
                pl.col("target_float").floor().cast(pl.Int64).alias("n_samples")
            )

            # Adjust sample counts due to flooring to match TARGET_SAMPLE_SIZE exactly
            samples_so_far = bin_counts["n_samples"].sum()
            difference = TARGET_SAMPLE_SIZE - samples_so_far
            if difference > 0:
                # Add the difference based on the fractional part, prioritizing larger fractions
                bin_counts = bin_counts.with_columns(
                    (pl.col("target_float") - pl.col("n_samples")).alias(
                        "fractional_part"
                    )
                )
                # Get indices of rows with the largest fractional parts to add the difference
                indices_to_increment = (
                    bin_counts.sort("fractional_part", descending=True)
                    .head(difference)
                    .select(pl.col("length_bin"))
                )

                bin_counts = bin_counts.with_columns(
                    pl.when(pl.col("length_bin").is_in(indices_to_increment))
                    .then(pl.col("n_samples") + 1)
                    .otherwise(pl.col("n_samples"))
                    .alias("n_samples")
                )

            samples_per_bin_map = {
                row["length_bin"]: row["n_samples"]
                for row in bin_counts.select(["length_bin", "n_samples"]).iter_rows(
                    named=True
                )
            }
            log.info(f"Calculated samples per bin: {samples_per_bin_map}")

            # Perform sampling within each bin
            sampled_ids_list = []
            for bin_label, n_to_sample in samples_per_bin_map.items():
                if n_to_sample > 0:
                    try:
                        sampled_ids_in_bin = (
                            proteome_df_filtered.filter(
                                pl.col("length_bin") == bin_label
                            )
                            .sample(
                                n=n_to_sample, seed=RANDOM_SEED, shuffle=False
                            )  # Explicitly set shuffle=False
                            .get_column(id_column_name)
                            .to_list()
                        )
                        sampled_ids_list.extend(sampled_ids_in_bin)
                    except Exception as sample_err:
                        log.warning(
                            f"Could not sample {n_to_sample} from bin {bin_label} (size {bin_counts.filter(pl.col('length_bin') == bin_label)['len'].item()}). Error: {sample_err}. Skipping bin."
                        )

            uniprot_ids_to_process = sampled_ids_list
            log.info(
                f"Selected {len(uniprot_ids_to_process)} proteins via stratified sampling for initial attempt."
            )
            # Check if the final count matches the target, log if not
            if len(uniprot_ids_to_process) != TARGET_SAMPLE_SIZE:
                log.warning(
                    f"Final selected sample size ({len(uniprot_ids_to_process)}) does not match target ({TARGET_SAMPLE_SIZE}) due to sampling issues or bin sizes."
                )


# Log the outcome
if proteome_df_filtered is not None and uniprot_ids_to_process:
    log.info(
        f"--- Data Loading and Sampling Complete: {len(uniprot_ids_to_process)} IDs selected for processing ---"
    )
    # Optional: Display first few rows of the filtered/binned dataframe
    # print(proteome_df_filtered.head())
elif not HUMAN_PROTEOME_FILE.exists():
    log.error("--- Data Loading Failed: Input file missing. ---")
else:
    log.error(
        "--- Data Loading and Sampling Failed: Check logs for errors. Cannot proceed. ---"
    )

# %% [markdown]
# ## 4. Helper Functions for Data Preparation
#
# Define functions to download AlphaFold structures and run DSSP for enrichment.
# These will be used in the next step.

# %%
import shutil
import requests
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import concurrent.futures
from tqdm.notebook import tqdm  # Import tqdm here for use in prepare_structures

# Ensure log is available (defined in cell 1)
if "log" not in locals():
    import logging

    log = logging.getLogger(__name__)
    if not log.hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_dssp_installed() -> bool:
    """Check if the mkdssp executable is available in the system PATH."""
    dssp_executable = "mkdssp"
    found = shutil.which(dssp_executable) is not None
    if not found:
        log.warning(
            f"'{dssp_executable}' not found in PATH. DSSP enrichment will be skipped."
        )
    return found


def download_alphafold_structure(
    uniprot_id: str, output_dir: Path, format: str = "cif"
) -> Optional[Path]:
    """Download AlphaFold structure for a given UniProt ID.

    Args:
        uniprot_id: UniProt identifier.
        output_dir: Directory to save structure file.
        format: File format ('cif' or 'pdb').

    Returns:
        Path to downloaded file or None if download failed. Raises exceptions on errors.
    """
    ext = format.lower()
    # Using the v4 model URL structure
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.{ext}"
    output_file = output_dir / f"AF-{uniprot_id}-F1-model_v4.{ext}"

    # Avoid re-downloading if file exists and is not empty
    if output_file.exists() and output_file.stat().st_size > 0:
        # log.debug(f"Raw file already exists for {uniprot_id}: {output_file}")
        return output_file

    # Use try...finally to ensure cleanup even if errors occur during download/write
    try:
        # log.debug(f"Attempting download for {uniprot_id} from {url}")
        response = requests.get(url, timeout=60)  # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(response.content)

        # Verify download wasn't empty
        if output_file.stat().st_size == 0:
            log.warning(f"Downloaded file for {uniprot_id} is empty. Deleting.")
            output_file.unlink(missing_ok=True)  # Safely delete if exists
            return None  # Return None for empty file, treat as failure
        # log.debug(f"Successfully downloaded {uniprot_id} to {output_file}")
        return output_file

    except requests.exceptions.HTTPError as http_err:
        # Handle 404 specifically as 'not found', re-raise others
        if response.status_code == 404:
            log.debug(f"Structure not found (404) for {uniprot_id} at {url}")
            return None  # Return None for 404, treat as failure
        else:
            log.warning(
                f"HTTP Error downloading {uniprot_id}: Status {response.status_code} for URL {url}. Error: {http_err}"
            )
            raise  # Re-raise other HTTP errors
    except Exception:
        # For any other download/IO error, re-raise
        raise
    finally:
        # Clean up potentially partial/empty file if an error occurred *after* file creation
        # but *before* successful return. Check if it exists and might be problematic.
        # This is a bit heuristic. A more robust way might involve temporary files.
        if output_file.exists():
            try:
                # If we are returning successfully, st_size should be > 0
                # If we return None (404 or empty file), we already handled unlink or it's correct
                # If we are raising an exception, we might have a partial file.
                # Check if size is 0 if an exception is being propagated.
                # This logic is tricky without knowing exactly when the exception occurred.
                # Simplest: assume if exception happened, file might be corrupt, remove if size is 0.
                if output_file.stat().st_size == 0:
                    log.debug(
                        f"Cleaning up zero-byte file after error for {uniprot_id}"
                    )
                    output_file.unlink(missing_ok=True)
            except Exception as cleanup_err:
                log.error(f"Error during cleanup for {uniprot_id}: {cleanup_err}")


def run_dssp(input_cif_path: Path, output_dir: Path) -> Optional[Path]:
    """Run DSSP on a structure file to add secondary structure information.

    Requires 'mkdssp' to be installed and in the system PATH.

    Args:
        input_cif_path: Path to the input structure file (CIF format).
        output_dir: Directory to save the DSSP-enriched CIF file.

    Returns:
        Path to enriched structure file or None if DSSP failed, wasn't installed, or input was invalid.
        Raises exceptions on unexpected errors.
    """
    # Double-check DSSP is installed before trying to run
    if not is_dssp_installed():
        # Warning is logged by is_dssp_installed if it returns False
        return None

    if (
        not input_cif_path
        or not input_cif_path.exists()
        or input_cif_path.stat().st_size == 0
    ):
        log.warning(f"DSSP input file missing, empty, or invalid: {input_cif_path}")
        return None

    # Define output path based on input stem
    output_file = output_dir / f"{input_cif_path.stem}.dssp.cif"

    # Skip if output file already exists, is not empty, and is newer than input
    if (
        output_file.exists()
        and output_file.stat().st_size > 0
        and output_file.stat().st_mtime >= input_cif_path.stat().st_mtime
    ):
        # log.debug(f"Enriched file already exists and is up-to-date for {input_cif_path.name}: {output_file}")
        return output_file

    # Use try...finally for cleanup
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # log.debug(f"Running DSSP on {input_cif_path.name} -> {output_file}")
        result = subprocess.run(
            ["mkdssp", str(input_cif_path), str(output_file)],
            check=True,  # Raise CalledProcessError on non-zero exit code
            capture_output=True,
            text=True,
            timeout=120,  # Add a timeout
        )
        # Check if output file was actually created and is not empty
        if not output_file.exists() or output_file.stat().st_size == 0:
            log.warning(
                f"DSSP ran for {input_cif_path.name} but output is missing or empty. Stderr: {result.stderr.strip()}"
            )
            if output_file.exists():
                output_file.unlink()  # Clean up empty file
            return None  # Treat as failure

        # log.debug(f"DSSP completed successfully for {input_cif_path.name}")
        return output_file

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as dssp_err:
        # Log specific, expected DSSP errors but return None (treat as failure)
        log.warning(f"DSSP execution failed for {input_cif_path.name}: {dssp_err}")
        return None
    except Exception:
        # Re-raise unexpected errors during DSSP execution
        raise
    finally:
        # If an exception occurred after file creation, check if it's problematic (e.g., empty)
        if output_file.exists():
            try:
                # Similar heuristic cleanup as in download function
                if output_file.stat().st_size == 0:
                    log.debug(
                        f"Cleaning up zero-byte DSSP output after error for {input_cif_path.name}"
                    )
                    output_file.unlink(missing_ok=True)
            except Exception as cleanup_err:
                log.error(
                    f"Error during DSSP cleanup for {input_cif_path.name}: {cleanup_err}"
                )


# %% [markdown]
# ## 5. Data Preparation: Download & Enrich Structures
#
# Iteratively download and enrich structures for the sampled UniProt IDs, handling failures and resampling if necessary to reach the target count.

# %%
import os  # Needed for os.cpu_count() inside prepare_structures if not imported earlier
import polars as pl  # Needed for resampling logic


def prepare_structures(
    uniprot_ids_initial_sample: List[str],
    proteome_df_full_binned: pl.DataFrame,
    target_total_success: int,
    num_bins: int,
    id_column_name: str,
    length_bin_column_name: str,
    raw_structure_dir: Path,
    enriched_structure_dir: Path,
    random_seed: int,
    num_workers: Optional[int] = None,
) -> Dict[str, Tuple[Path, Optional[str]]]:
    """Downloads and enriches structures, resampling from bins on failure to reach target.

    Attempts to process the initial sample. If the target number of successful
    preparations (download + optional DSSP) is not met, it iteratively
    resamples additional candidates from the appropriate length bins in the full
    dataset until the target is reached or candidates are exhausted.

    Checks for existing files before submitting download/DSSP tasks. Handles errors
    within individual download/dssp tasks gracefully.

    Args:
        uniprot_ids_initial_sample: List of UniProt IDs from initial stratified sample.
        proteome_df_full_binned: Polars DataFrame with all potential proteins, including
                                   UniProt IDs (id_column_name) and length bins
                                   (length_bin_column_name).
        target_total_success: The desired number of successfully prepared structures.
        num_bins: The number of bins used for stratification (only for logging/context).
        id_column_name: Name of the column containing UniProt IDs in the DataFrame.
        length_bin_column_name: Name of the column containing length bin labels.
        raw_structure_dir: Directory for raw downloaded CIF files.
        enriched_structure_dir: Directory for DSSP-enriched CIF files.
        random_seed: Seed for reproducible resampling.
        num_workers: Number of parallel workers. Defaults to os.cpu_count().

    Returns:
        A dictionary mapping successfully prepared UniProt IDs to a tuple:
        - Path: Path to the *final* structure file (enriched if DSSP ran, raw otherwise).
        - Optional[str]: Error message if preparation failed, None otherwise.
        The dictionary size will be <= target_total_success.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    log.info(
        f"--- Starting Iterative Structure Preparation (Target: {target_total_success}) ---"
    )
    log.info(f"Using {num_workers} workers for parallel tasks.")

    # Ensure parent directories exist (safer than assuming they do)
    raw_structure_dir.mkdir(parents=True, exist_ok=True)
    enriched_structure_dir.mkdir(parents=True, exist_ok=True)

    final_results: Dict[
        str, Tuple[Path, Optional[str]]
    ] = {}  # Store final path and error status
    attempted_ids = set(uniprot_ids_initial_sample)  # Track all IDs ever considered
    ids_to_process_queue = list(
        uniprot_ids_initial_sample
    )  # Current queue of IDs to work on

    # --- Calculate Target Successes Per Bin (for balanced resampling) ---
    # Filter the main dataframe to only include the initial sample IDs
    initial_sample_df = proteome_df_full_binned.filter(
        pl.col(id_column_name).is_in(uniprot_ids_initial_sample)
    )
    initial_bin_counts = initial_sample_df.group_by(length_bin_column_name).len()

    # Calculate proportional target, ensuring total matches target_total_success
    target_success_per_bin = {}
    float_targets = {}
    if initial_bin_counts.height > 0 and len(uniprot_ids_initial_sample) > 0:
        total_initial = len(uniprot_ids_initial_sample)
        for row in initial_bin_counts.iter_rows(named=True):
            bin_label = row[length_bin_column_name]
            proportion = row["len"] / total_initial
            target_float = proportion * target_total_success
            target_success_per_bin[bin_label] = int(target_float)  # Floor initially
            float_targets[bin_label] = target_float

        # Adjust for rounding errors to exactly match target_total_success
        current_total_target = sum(target_success_per_bin.values())
        diff = target_total_success - current_total_target
        if diff != 0:
            # Sort bins by the fractional part of their target to distribute the difference
            sorted_bins = sorted(
                target_success_per_bin.keys(),
                key=lambda k: float_targets[k] - target_success_per_bin[k],
                reverse=True,  # Prioritize bins with larger fractions
            )
            for i in range(diff):  # Add the difference 1 by 1
                bin_to_adjust = sorted_bins[i % len(sorted_bins)]
                target_success_per_bin[bin_to_adjust] += 1

        log.info(f"Target successful preparations per bin: {target_success_per_bin}")
    else:
        log.warning(
            "Could not calculate target per bin (no initial sample or bins). Resampling will be random if needed."
        )
        # Fallback: aim for total target without bin balancing if calculation fails
        target_success_per_bin = None

    current_success_per_bin = (
        {bin_label: 0 for bin_label in target_success_per_bin}
        if target_success_per_bin
        else {}
    )

    # --- Main Processing Loop ---
    pbar_main = tqdm(
        total=target_total_success,
        desc="Preparing Structures",
        unit="struct",
        leave=True,
    )
    dssp_was_checked = False
    dssp_available = False  # Check only once

    while len(final_results) < target_total_success and ids_to_process_queue:
        if not dssp_was_checked:
            dssp_available = is_dssp_installed()  # Check DSSP availability
            dssp_was_checked = True

        # Process in batches for efficiency
        batch_to_process = ids_to_process_queue[: num_workers * 5]
        ids_to_process_queue = ids_to_process_queue[num_workers * 5 :]
        log.info(
            f"Processing batch of {len(batch_to_process)} IDs. Queue: {len(ids_to_process_queue)}. Successes: {len(final_results)}/{target_total_success}"
        )

        batch_needs_download: List[str] = []
        batch_needs_dssp: Dict[str, Path] = {}  # uniprot_id -> raw_file_path
        batch_pre_checked: Dict[
            str, Tuple[Path, Optional[str]]
        ] = {}  # Store results found during pre-check

        # 1. Pre-check Batch: Look for existing raw/enriched files
        for uniprot_id in batch_to_process:
            raw_file = raw_structure_dir / f"AF-{uniprot_id}-F1-model_v4.cif"
            enriched_file = (
                enriched_structure_dir / f"AF-{uniprot_id}-F1-model_v4.dssp.cif"
            )

            # Priority: Check for valid enriched file if DSSP is available
            if (
                dssp_available
                and enriched_file.exists()
                and enriched_file.stat().st_size > 0
            ):
                # Check if it's newer than raw (if raw exists)
                needs_regen = False
                if (
                    raw_file.exists()
                    and enriched_file.stat().st_mtime < raw_file.stat().st_mtime
                ):
                    log.debug(
                        f"Enriched file for {uniprot_id} is older than raw. Will regenerate."
                    )
                    needs_regen = True

                if not needs_regen:
                    batch_pre_checked[uniprot_id] = (
                        enriched_file,
                        None,
                    )  # Found valid enriched
                    continue  # Skip further checks for this ID in this batch

            # Check for valid raw file
            if raw_file.exists() and raw_file.stat().st_size > 0:
                if dssp_available:
                    batch_needs_dssp[uniprot_id] = (
                        raw_file  # Raw exists, needs (re)generation
                    )
                else:
                    batch_pre_checked[uniprot_id] = (
                        raw_file,
                        None,
                    )  # Raw exists, DSSP not available/needed
                continue

            # If neither exists, needs download
            batch_needs_download.append(uniprot_id)

        # 2. Download Phase (concurrently)
        download_results_batch: Dict[
            str, Optional[Path]
        ] = {}  # uniprot_id -> path or None
        if batch_needs_download:
            log.info(f"Attempting download for {len(batch_needs_download)} IDs...")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="dl_"
            ) as executor:
                future_to_id_dl = {
                    executor.submit(
                        download_alphafold_structure, uniprot_id, raw_structure_dir
                    ): uniprot_id
                    for uniprot_id in batch_needs_download
                }
                for future in concurrent.futures.as_completed(future_to_id_dl):
                    uniprot_id = future_to_id_dl[future]
                    try:
                        file_path = (
                            future.result()
                        )  # Will be None on 404/empty, raises on other errors
                        download_results_batch[uniprot_id] = file_path
                        if file_path and dssp_available:
                            batch_needs_dssp[uniprot_id] = (
                                file_path  # Add successful download to DSSP list
                            )
                        elif file_path:  # Download ok, but no DSSP
                            batch_pre_checked[uniprot_id] = (file_path, None)
                        # else: download failed (404/empty), will be handled later
                    except Exception as exc:
                        # Log errors from download_alphafold_structure if they weren't handled (i.e., not 404/empty)
                        log.error(
                            f"Download task for {uniprot_id} generated an exception: {exc}",
                            exc_info=False,  # Don't print traceback if helper already logged
                        )
                        download_results_batch[uniprot_id] = (
                            None  # Explicitly mark as failed
                        )

        # 3. Enrich Phase (concurrently for those needing it)
        dssp_results_batch: Dict[
            str, Optional[Path]
        ] = {}  # uniprot_id -> enriched_path or None
        if batch_needs_dssp and dssp_available:
            log.info(f"Attempting DSSP enrichment for {len(batch_needs_dssp)} IDs...")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="dssp_"
            ) as executor:
                future_to_id_dssp = {
                    executor.submit(
                        run_dssp, raw_path, enriched_structure_dir
                    ): uniprot_id
                    for uniprot_id, raw_path in batch_needs_dssp.items()
                }
                for future in concurrent.futures.as_completed(future_to_id_dssp):
                    uniprot_id = future_to_id_dssp[future]
                    try:
                        enriched_path = (
                            future.result()
                        )  # Will be None on failure, raises on unexpected errors
                        dssp_results_batch[uniprot_id] = enriched_path
                    except Exception as exc:
                        log.error(
                            f"DSSP task for {uniprot_id} generated an exception: {exc}",
                            exc_info=False,  # Don't print traceback if helper already logged
                        )
                        dssp_results_batch[uniprot_id] = (
                            None  # Explicitly mark as failed
                        )

        # 4. Process Batch Results and Update Global State
        newly_successful_count = 0
        for uniprot_id in batch_to_process:
            # Skip if already added to final results in a previous batch
            if uniprot_id in final_results:
                continue

            final_path: Optional[Path] = None
            error_msg: Optional[str] = None
            processed = False  # Flag to track if we handled this ID

            # Check pre-checked results first
            if uniprot_id in batch_pre_checked:
                final_path, error_msg = batch_pre_checked[uniprot_id]
                processed = True

            # Check DSSP results (if applicable)
            if not processed and uniprot_id in dssp_results_batch:
                enriched_path = dssp_results_batch[uniprot_id]
                if enriched_path:
                    final_path = enriched_path
                    error_msg = None
                else:
                    error_msg = "dssp_failed"
                    # Try to fallback to raw file if DSSP failed but download succeeded
                    raw_path_from_dssp_input = batch_needs_dssp.get(uniprot_id)
                    # Check if raw_path exists (was downloaded or pre-existed)
                    raw_path_exists = (
                        raw_path_from_dssp_input and raw_path_from_dssp_input.exists()
                    )
                    # Also check download didn't fail explicitly
                    download_succeeded = (
                        uniprot_id in download_results_batch
                        and download_results_batch[uniprot_id] is not None
                    )

                    if raw_path_exists and (
                        uniprot_id not in download_results_batch or download_succeeded
                    ):
                        log.warning(
                            f"DSSP failed for {uniprot_id}, but raw file exists. Using raw file."
                        )
                        final_path = raw_path_from_dssp_input
                        error_msg = None  # Reset error as we have a usable file
                    else:
                        # DSSP failed and no fallback raw file found/valid in this context
                        final_path = None  # Ensure no path is set

                processed = True

            # Check download results (if not handled by pre-check or DSSP)
            if not processed and uniprot_id in download_results_batch:
                raw_path = download_results_batch[uniprot_id]
                if raw_path:
                    # This case implies DSSP wasn't run or needed (or failed but we are falling back)
                    final_path = raw_path
                    error_msg = None
                else:
                    final_path = None
                    error_msg = "download_failed"  # Includes 404/empty file case now
                processed = True

            # If still not processed, it means it wasn't in any list (shouldn't happen ideally)
            if not processed:
                log.error(
                    f"ID {uniprot_id} was in batch but not found in any result dict. Marking as unknown failure."
                )
                final_path = None
                error_msg = "unknown_prep_failure"

            # --- Update Success Counts and Final Results ---
            if final_path is not None and error_msg is None:
                # Successfully prepared! Check if bin target is met.
                can_add = True
                if target_success_per_bin:  # Only check bins if calculated
                    try:
                        # Get bin label for this ID (requires the dataframe)
                        bin_label = (
                            proteome_df_full_binned.filter(
                                pl.col(id_column_name) == uniprot_id
                            )
                            .select(length_bin_column_name)
                            .item()
                        )  # Get single value

                        if (
                            current_success_per_bin[bin_label]
                            >= target_success_per_bin[bin_label]
                        ):
                            log.debug(
                                f"Bin '{bin_label}' target met ({target_success_per_bin[bin_label]}). Skipping adding extra success {uniprot_id}."
                            )
                            can_add = False
                        # else: log.debug(f"Adding {uniprot_id} to bin {bin_label} ({current_success_per_bin[bin_label]+1}/{target_success_per_bin[bin_label]})")

                    except Exception as e:
                        log.error(
                            f"Error retrieving bin for successful ID {uniprot_id}: {e}. Adding to results without bin check."
                        )
                        # Decide if you want to add it anyway or skip on error
                        # can_add = False

                if can_add and len(final_results) < target_total_success:
                    final_results[uniprot_id] = (final_path, None)
                    newly_successful_count += 1
                    if target_success_per_bin:
                        current_success_per_bin[bin_label] += (
                            1  # Increment bin count only if added
                        )
                    pbar_main.update(1)  # Update progress bar for each success added
                elif len(final_results) >= target_total_success:
                    # Stop adding if overall target met, even if bin had space
                    log.debug(
                        f"Overall target {target_total_success} reached. Not adding further success {uniprot_id}."
                    )

            else:
                # Preparation failed for this ID
                log.debug(f"Preparation failed for {uniprot_id}: {error_msg}")
                # Optionally store failure reason if needed elsewhere
                # final_results[uniprot_id] = (None, error_msg)
                pass  # Don't add failures to the final success dict

            # Add to attempted_ids regardless of success/failure for resampling logic
            attempted_ids.add(uniprot_id)

        # log.info(f"Batch processed. Newly successful: {newly_successful_count}. Total successful: {len(final_results)}")

        # 5. Resampling Logic: If queue is empty and target not met
        if not ids_to_process_queue and len(final_results) < target_total_success:
            log.info("Processing queue empty. Checking if resampling is needed...")
            needed_overall = target_total_success - len(final_results)
            resample_candidates = []

            # Determine how many are needed per bin if using bin targets
            needed_per_bin = {}
            if target_success_per_bin:
                for bin_label in target_success_per_bin:
                    needed = target_success_per_bin[
                        bin_label
                    ] - current_success_per_bin.get(bin_label, 0)
                    if needed > 0:
                        needed_per_bin[bin_label] = needed

            if needed_per_bin:  # Resample based on bin needs
                log.info(f"Bins needing structures: {needed_per_bin}")
                for bin_label, needed_count in needed_per_bin.items():
                    # Find candidates in this bin not yet attempted
                    candidates_in_bin = proteome_df_full_binned.filter(
                        (pl.col(length_bin_column_name) == bin_label)
                        & (
                            ~pl.col(id_column_name).is_in(list(attempted_ids))
                        )  # Exclude already tried IDs
                    )
                    n_available = candidates_in_bin.height
                    if n_available > 0:
                        # Resample slightly more than needed (e.g., 2x) to account for future failures, but not more than available
                        n_to_sample = min(needed_count * 2, n_available)
                        log.info(
                            f"Bin {bin_label}: Needs {needed_count}, Available {n_available}. Resampling {n_to_sample} new candidates."
                        )
                        try:
                            new_samples = (
                                candidates_in_bin.sample(
                                    n=n_to_sample, seed=random_seed, shuffle=False
                                )  # Explicitly set shuffle=False
                                .get_column(id_column_name)
                                .to_list()
                            )
                            resample_candidates.extend(new_samples)
                            attempted_ids.update(
                                new_samples
                            )  # Mark these as attempted now
                        except Exception as sample_err:
                            log.warning(
                                f"Resampling from bin {bin_label} failed: {sample_err}. Skipping bin for this round."
                            )
                    else:
                        log.warning(
                            f"No more candidates available in bin {bin_label} to meet target."
                        )

            elif (
                needed_overall > 0
            ):  # Fallback: Resample randomly if bin targets not used/calculated
                log.info(
                    f"Resampling randomly to fulfill remaining {needed_overall} target."
                )
                # Find all candidates not yet attempted
                all_candidates = proteome_df_full_binned.filter(
                    ~pl.col(id_column_name).is_in(list(attempted_ids))
                )
                n_available = all_candidates.height
                if n_available > 0:
                    n_to_sample = min(needed_overall * 2, n_available)
                    log.info(
                        f"Available {n_available}. Resampling {n_to_sample} new candidates."
                    )
                    try:
                        new_samples = (
                            all_candidates.sample(
                                n=n_to_sample, seed=random_seed, shuffle=False
                            )  # Explicitly set shuffle=False
                            .get_column(id_column_name)
                            .to_list()
                        )
                        resample_candidates.extend(new_samples)
                        attempted_ids.update(new_samples)
                    except Exception as sample_err:
                        log.warning(f"Random resampling failed: {sample_err}.")
                else:
                    log.warning(
                        "No more candidates available anywhere for random resampling."
                    )

            # Add new candidates to the queue if any were found
            if resample_candidates:
                log.info(
                    f"Adding {len(resample_candidates)} resampled candidates to the processing queue."
                )
                ids_to_process_queue.extend(resample_candidates)
            else:
                # No more candidates could be found anywhere
                log.warning(
                    f"Target of {target_total_success} not met. No more candidates found for resampling. "
                    f"Achieved {len(final_results)} successful preparations."
                )
                break  # Exit the main while loop

    # --- Loop End ---
    pbar_main.close()
    final_success_count = len(final_results)
    log.info(f"--- Structure Preparation Finished ---")
    log.info(
        f"Successfully prepared: {final_success_count} / {target_total_success} target."
    )
    log.info(f"Total unique IDs attempted: {len(attempted_ids)}")
    if final_success_count < target_total_success:
        log.warning(f"Could only prepare {final_success_count} structures.")

    # Filter out entries with errors before returning
    successful_results = {
        uid: (path, None)
        for uid, (path, err) in final_results.items()
        if err is None and path is not None
    }

    return successful_results


# --- Execute the Preparation ---
# Ensure previous cell variables are available
if (
    "uniprot_ids_to_process" in locals()
    and "proteome_df_filtered" in locals()
    and proteome_df_filtered is not None
    and uniprot_ids_to_process
):
    # Make sure directories are created (redundant if done in cell 1, but safe)
    raw_structures_dir.mkdir(parents=True, exist_ok=True)
    enriched_structures_dir.mkdir(parents=True, exist_ok=True)

    prepared_structures_results: Dict[str, Tuple[Path, Optional[str]]] = (
        prepare_structures(
            uniprot_ids_initial_sample=uniprot_ids_to_process,
            proteome_df_full_binned=proteome_df_filtered,
            target_total_success=TARGET_SAMPLE_SIZE,
            num_bins=NUM_BINS,
            id_column_name=id_column_name,  # Should be defined from cell 3
            length_bin_column_name="length_bin",  # Name assigned during binning in cell 3
            raw_structure_dir=raw_structures_dir,
            enriched_structure_dir=enriched_structures_dir,
            random_seed=RANDOM_SEED,
            num_workers=NUM_WORKERS,
        )
    )
else:
    log.error(
        "Required variables (uniprot_ids_to_process, proteome_df_filtered) not available from previous steps. Skipping structure preparation."
    )
    prepared_structures_results = {}  # Initialize as empty dict

# %% [markdown]
# ## 6. Save Final Prepared IDs and Lengths
#
# After preparation, parse the length from each successfully prepared structure file and save the list of UniProt IDs and their lengths to a TSV file. This list represents the actual structures that will be used in the benchmark.

# %%
import polars as pl
from flatprot.io import GemmiStructureParser
from flatprot.core import Structure

log.info("--- Saving Final Prepared IDs and Lengths ---")

final_ids_data = []

if "prepared_structures_results" in locals() and prepared_structures_results:
    log.info(
        f"Processing {len(prepared_structures_results)} successfully prepared structures to extract lengths..."
    )
    parser = GemmiStructureParser()
    skipped_parsing = 0

    for uniprot_id, (structure_file_path, _) in tqdm(
        prepared_structures_results.items(), desc="Extracting Lengths", unit="struct"
    ):
        if structure_file_path and structure_file_path.exists():
            try:
                # Parse the structure to get the length
                structure_obj: Structure = parser.parse_structure(structure_file_path)
                length = len(structure_obj.residues)
                if length > 0:
                    final_ids_data.append(
                        {
                            "uniprot_id": uniprot_id,
                            "length": length,
                            "file_path": str(structure_file_path),
                        }
                    )
                else:
                    log.warning(
                        f"Parsed structure for {uniprot_id} has length 0. Skipping."
                    )
                    skipped_parsing += 1
            except Exception as e:
                log.error(
                    f"Failed to parse structure file {structure_file_path} for length. Skipping ID {uniprot_id}. Error: {e}"
                )
                skipped_parsing += 1
        else:
            log.warning(
                f"Structure file path missing or does not exist for {uniprot_id}. Skipping."
            )
            skipped_parsing += 1

    if final_ids_data:
        try:
            final_ids_df = pl.DataFrame(final_ids_data)
            # Ensure results directory exists
            SAMPLED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            final_ids_df.write_csv(
                SAMPLED_IDS_FILE, separator="\t", include_header=True
            )
            log.info(
                f"Saved {len(final_ids_data)} final UniProt IDs and lengths to: {SAMPLED_IDS_FILE}"
            )
            if skipped_parsing > 0:
                log.warning(
                    f"Skipped {skipped_parsing} entries during length extraction due to parsing errors or missing files."
                )
        except Exception as e:
            log.error(
                f"Failed to write final sampled IDs TSV to {SAMPLED_IDS_FILE}: {e}",
                exc_info=True,
            )
    else:
        log.warning("No valid data to save for final sampled IDs.")

else:
    log.warning(
        "No prepared structures found ('prepared_structures_results' dictionary is empty or missing). Cannot save final IDs list."
    )

log.info("--- Finished Saving Final IDs ---")


# %% [markdown]
# ## 7. Helper Functions for Benchmarking
#
# Define functions to perform the inertia-based and alignment-based projections/renderings, and a function to save results.

# %%
import time
import shutil
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import polars as pl  # For saving results

# Import necessary FlatProt components
from flatprot.core import Structure
from flatprot.io import GemmiStructureParser
from flatprot.utils.structure_utils import (
    transform_structure_with_inertia,
    project_structure_orthographically,
)
from flatprot.utils.scene_utils import create_scene_from_structure
from flatprot.renderers import SVGRenderer
from flatprot.alignment import (
    align_structure_database,
    AlignmentResult,
    NoSignificantAlignmentError,
)
from flatprot.transformation import (
    MatrixTransformParameters,
    MatrixTransformer,
)
from flatprot.core.errors import FlatProtError
from flatprot.utils.database import ensure_database_available, DEFAULT_DB_DIR

# Ensure log and canvas dimensions are available (defined in earlier cells)
if "log" not in locals():
    import logging

    log = logging.getLogger(__name__)
    if not log.hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
if "CANVAS_WIDTH" not in locals():
    CANVAS_WIDTH = 1000  # Default fallback
if "CANVAS_HEIGHT" not in locals():
    CANVAS_HEIGHT = 1000  # Default fallback


def benchmark_inertia_projection_svg(
    structure_file: Path, parser: GemmiStructureParser
) -> Tuple[float, int]:
    """Benchmarks inertia transform, orthographic projection, and SVG rendering.

    Args:
        structure_file: Path to the (enriched) structure file.
        parser: An instance of GemmiStructureParser.

    Returns:
        A tuple containing:
        - float: Time for parse, transform, projection, scene, render (seconds).
        - int: Protein length (number of residues).
        Raises exceptions on failure.
    """
    start_time = time.perf_counter()

    # --- Parse ---
    structure_obj: Structure = parser.parse_structure(structure_file)
    length = len(structure_obj.residues)
    if length == 0:
        raise ValueError("Parsed structure has 0 residues.")

    # --- Transform (Inertia) ---
    transformed_structure = transform_structure_with_inertia(structure_obj)

    # --- Project (Orthographic) ---
    projected_structure = project_structure_orthographically(
        transformed_structure,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        maintain_aspect_ratio=True,
        center_projection=True,
    )

    # --- Create Scene ---
    scene = create_scene_from_structure(projected_structure)

    # --- Render (SVG) ---
    renderer = SVGRenderer(scene, CANVAS_WIDTH, CANVAS_HEIGHT)
    _ = renderer.get_svg_string()  # Generate the string

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time, length


def benchmark_align_project_svg(
    structure_file: Path,
    parser: GemmiStructureParser,
    db_path: Path,  # This should be the Foldseek DB index path itself
    foldseek_path: str,
    min_probability: float,
) -> Tuple[float, int, bool]:
    """Benchmarks read, align (Foldseek), project with alignment, and render SVG.

    Args:
        structure_file: Path to the (enriched) structure file.
        parser: An instance of GemmiStructureParser.
        db_path: Path to FoldSeek DB index file (e.g., .../db).
        foldseek_path: Path to FoldSeek executable.
        min_probability: Minimum alignment probability threshold.

    Returns:
        Tuple: (time_seconds, length, has_match)
        Raises exceptions on failure, except for NoSignificantAlignmentError.
    """
    length = 0
    has_match = False
    try:
        start_time = time.perf_counter()

        # --- Parse ---
        structure_obj: Structure = parser.parse_structure(structure_file)
        length = len(structure_obj.residues)
        if length == 0:
            raise ValueError("Parsed structure has 0 residues.")

        # --- Align (Foldseek) ---
        alignment_result: AlignmentResult = align_structure_database(
            structure_file=structure_file,
            foldseek_db_path=db_path,  # Pass the specific DB index path
            foldseek_command=foldseek_path,
            min_probability=min_probability,
            target_db_id=None,  # Find best match in the database
        )
        has_match = True  # Reaching here means alignment succeeded above threshold
        rotation_matrix = alignment_result.rotation_matrix

        # --- Transform (Alignment Matrix) ---
        transformer_params = MatrixTransformParameters(matrix=rotation_matrix)
        transformer = MatrixTransformer(parameters=transformer_params)
        transformed_structure = structure_obj.apply_vectorized_transformation(
            lambda coords: transformer.transform(coords, arguments=None)
        )

        # --- Project (Orthographic) ---
        projected_structure = project_structure_orthographically(
            transformed_structure,
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
            maintain_aspect_ratio=True,
            center_projection=True,
        )

        # --- Create Scene ---
        scene = create_scene_from_structure(projected_structure)

        # --- Render (SVG) ---
        renderer = SVGRenderer(scene, CANVAS_WIDTH, CANVAS_HEIGHT)
        _ = renderer.get_svg_string()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return elapsed_time, length, has_match

    except NoSignificantAlignmentError:
        # This is a valid outcome, not an error. Record time taken.
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        has_match = False
        # log.debug(f"No significant alignment found for {structure_file.name}.")
        return (
            elapsed_time,
            length if length else 0,  # Return length if parsed before error
            has_match,
        )  # Return time, length, False match


def save_results(results_batch: List[Dict], output_file: Path) -> None:
    """Saves a batch of benchmark results to a single TSV file.

    Creates file/header if needed.

    Args:
        results_batch: List of dictionaries, each representing a benchmark run.
        output_file: Path to the output TSV file.
    """
    if not results_batch:
        return

    # Define schema for consistency
    schema = {
        "uniprot_id": pl.Utf8,
        "length": pl.Int64,
        "method": pl.Utf8,
        "runtime": pl.Float64,
        "has_match": pl.Boolean,
        "timestamp": pl.Utf8,
        "file_path": pl.Utf8,
        "error": pl.Utf8,
    }

    try:
        df_new = pl.DataFrame(results_batch, schema_overrides=schema)

        # Ensure results directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the entire DataFrame, overwriting if exists (Polars default)
        # Polars handles header creation automatically on first write
        df_new.write_csv(output_file, separator="\t", include_header=True)

        log.info(f"Saved {len(results_batch)} results to {output_file}")
    except Exception as e:
        log.error(f"Failed to save results batch to {output_file}: {e}", exc_info=True)


# %% [markdown]
# ## 8. Run Benchmarks
#
# Execute the defined benchmarks on the prepared structures (loaded from the file saved in step 6) and save results incrementally.

# %%
from tqdm.notebook import tqdm  # Ensure tqdm is available


def run_benchmarks(
    prepared_ids_file: Path,  # Input file with IDs, lengths, paths
    results_output_file: Path,  # Added back
    foldseek_path: str,
    min_probability: float,
    foldseek_db_parent_dir: Optional[Path] = None,  # Optional override for DB location
) -> None:
    """Runs the benchmark suite on prepared structures loaded from a file and saves results to a single file.

    Args:
        prepared_ids_file: Path to the TSV file containing 'uniprot_id', 'length', 'file_path'.
        results_output_file: Path to the TSV file for saving benchmark results. # Added back
        foldseek_path: Path to FoldSeek executable.
        min_probability: Minimum alignment probability threshold.
        foldseek_db_parent_dir: Path to the PARENT directory of the Foldseek database.
                                Uses default cache location if None.
    """
    log.info(f"--- Starting Benchmarks ---")
    if not prepared_ids_file.exists():
        log.error(
            f"Input file with prepared IDs not found: {prepared_ids_file}. Cannot run benchmarks."
        )
        return

    # --- Load Prepared Structures Data ---
    try:
        prepared_df = pl.read_csv(
            prepared_ids_file,
            separator="\t",
            has_header=True,
        )
        # Ensure required columns exist
        required_cols = ["uniprot_id", "length", "file_path"]
        if not all(col in prepared_df.columns for col in required_cols):
            raise ValueError(
                f"Input file {prepared_ids_file} missing required columns ({required_cols}). Found: {prepared_df.columns}"
            )
        log.info(
            f"Loaded {prepared_df.height} entries from {prepared_ids_file} for benchmarking."
        )
    except Exception as e:
        log.error(
            f"Failed to load or process prepared IDs file {prepared_ids_file}: {e}",
            exc_info=True,
        )
        return

    # --- Pre-checks for External Dependencies ---
    foldseek_valid = shutil.which(foldseek_path) or Path(foldseek_path).exists()
    if not foldseek_valid:
        log.error(
            f"Foldseek executable not found or not executable: {foldseek_path}. Alignment benchmark will be skipped."
        )

    # --- Ensure Foldseek Database is Available ---
    effective_db_path: Optional[Path] = None  # Parent dir of DB
    foldseek_db_index_path: Optional[Path] = None  # Actual target for foldseek command
    db_available = False
    try:
        log.info(
            "Ensuring alignment database is available (will download/check default if needed)..."
        )
        # Pass the user-provided parent dir OR None (to use default)
        effective_db_path = ensure_database_available(
            foldseek_db_parent_dir or DEFAULT_DB_DIR
        )
        # Construct the path to the actual Foldseek DB index file
        foldseek_db_index_path = effective_db_path / "foldseek" / "db"
        if foldseek_db_index_path.exists():
            log.info(f"Using Foldseek DB at: {foldseek_db_index_path}")
            db_available = True
        else:
            # This shouldn't happen if ensure_database_available worked, but check defensively
            log.error(
                f"Foldseek DB index file not found at {foldseek_db_index_path} after check/download attempt. Alignment benchmark skipped."
            )
            effective_db_path = None  # Reset path if index is missing
    except RuntimeError as db_err:
        log.error(
            f"Error obtaining Foldseek DB: {db_err}. Alignment benchmark skipped."
        )
        effective_db_path = None

    alignment_possible = (
        foldseek_valid and db_available and foldseek_db_index_path is not None
    )
    if not alignment_possible:
        log.warning(
            "Alignment benchmark cannot run due to missing Foldseek executable or database issues."
        )

    # --- Initialize Parser and Results List ---
    parser = GemmiStructureParser()
    all_results: List[Dict] = []  # Collect all results here
    processed_count = 0
    total_to_process = prepared_df.height

    # --- Main Benchmark Loop ---
    for row in tqdm(
        prepared_df.iter_rows(named=True),
        total=total_to_process,
        desc="Running Benchmarks",
        unit="structure",
    ):
        uniprot_id: str = row["uniprot_id"]
        # length_from_file: int = row["length"] # Length is re-calculated during benchmark for verification
        structure_file: Path = Path(row["file_path"])
        timestamp = datetime.now().isoformat()
        structure_file_str = str(structure_file)

        # Check if file exists before attempting benchmarks
        if not structure_file or not structure_file.exists():
            log.warning(
                f"Structure file not found for {uniprot_id} at path: {structure_file}. Skipping benchmarks."
            )
            all_results.append(
                {
                    "uniprot_id": uniprot_id,
                    "length": 0,
                    "method": "N/A",
                    "runtime": -1.0,
                    "has_match": False,
                    "timestamp": timestamp,
                    "file_path": structure_file_str,
                    "error": "input_file_missing",
                }
            )
            processed_count += 1
            continue  # Skip to next structure

        proj_time = -1.0
        proj_length = 0
        proj_err = None
        # --- Benchmark 1: Inertia + Projection + SVG ---
        try:
            proj_time, proj_length = benchmark_inertia_projection_svg(
                structure_file, parser
            )
        except Exception as e:
            log.warning(f"Inertia benchmark failed for {structure_file.name}: {e}")
            proj_err = f"{type(e).__name__}: {str(e)}"
            # Use length from file if parsing failed early, otherwise keep 0
            proj_length = row.get("length", 0) if proj_length == 0 else proj_length

        all_results.append(
            {
                "uniprot_id": uniprot_id,
                "length": proj_length,
                "method": "inertia",
                "runtime": proj_time,
                "has_match": False,
                "timestamp": timestamp,  # has_match not applicable
                "file_path": structure_file_str,
                "error": proj_err,
            }
        )

        align_time = -1.0
        align_length = proj_length  # Use length from inertia by default
        has_match = False
        align_err = None
        # --- Benchmark 2: Alignment + Projection + SVG ---
        if alignment_possible and foldseek_db_index_path:  # Ensure vars are valid
            try:
                align_time, align_length, has_match = benchmark_align_project_svg(
                    structure_file=structure_file,
                    parser=parser,
                    db_path=foldseek_db_index_path,  # Pass the actual index path
                    foldseek_path=foldseek_path,
                    min_probability=min_probability,
                )
            except Exception as e:
                # This block now catches all errors *except* NoSignificantAlignmentError handled inside the helper
                log.warning(
                    f"Alignment benchmark failed for {structure_file.name}: {e}"
                )
                align_err = f"{type(e).__name__}: {str(e)}"
                # Use length from inertia if align failed, otherwise keep value from benchmark func
                align_length = proj_length if align_length == 0 else align_length

            # Verify length matches if both benchmarks seemed to run okay structurally
            if proj_err is None and align_err is None and proj_length != align_length:
                log.warning(
                    f"Length mismatch for {uniprot_id}: Inertia ({proj_length}), Align ({align_length}). Using align length."
                )
                # Keep align_length as it's likely more accurate if both ran

            all_results.append(
                {
                    "uniprot_id": uniprot_id,
                    "length": align_length,  # Use align_length determined above
                    "method": "family",
                    "runtime": align_time,
                    "has_match": has_match,
                    "timestamp": timestamp,
                    "file_path": structure_file_str,
                    "error": align_err,
                }
            )
        else:
            # Record skipped alignment benchmark
            all_results.append(
                {
                    "uniprot_id": uniprot_id,
                    "length": proj_length,  # Use length from inertia if available
                    "method": "family",
                    "runtime": -1.0,
                    "has_match": False,
                    "timestamp": timestamp,
                    "file_path": structure_file_str,
                    "error": "alignment_skipped",
                }
            )

        processed_count += 1

    # --- Save all results after the loop ---
    if all_results:
        save_results(all_results, results_output_file)  # Pass output file
        log.info(f"Saved results for {len(all_results)} records.")
    else:
        log.warning("No benchmark results generated to save.")

    log.info(
        f"--- Benchmark Execution Finished. Results saved to: {results_output_file} ---"  # Updated log message
    )


# --- Execute the Benchmarks ---
# Ensure the input file from step 6 exists and config variables are set
if "SAMPLED_IDS_FILE" in locals() and SAMPLED_IDS_FILE.exists():
    run_benchmarks(
        prepared_ids_file=SAMPLED_IDS_FILE,
        results_output_file=RESULTS_FILE,  # Added back
        foldseek_path=FOLDSEEK_EXECUTABLE,  # Defined in cell 2
        min_probability=MIN_ALIGN_PROBABILITY,  # Defined in cell 2
        foldseek_db_parent_dir=FOLDSEEK_DB_PATH,  # Defined in cell 2 (can be None)
    )
elif "SAMPLED_IDS_FILE" in locals():
    log.error(f"Prepared IDs file ({SAMPLED_IDS_FILE}) not found. Run previous steps.")
else:
    log.error("Configuration variables missing. Cannot run benchmarks.")

# %% [markdown]
# ## 9. Plotting Results
#
# Load the final benchmark results and generate a log-log plot of runtime vs. protein length.

# %%
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
from pathlib import Path
import numpy as np  # For log calculations

# Ensure log and results_dir are available (defined in earlier cells)
if "log" not in locals():
    import logging

    log = logging.getLogger(__name__)
    if not log.hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
if "results_dir" not in locals():
    # Attempt to reconstruct results_dir if missing (adjust path if necessary)
    if "base_dir" in locals():
        results_dir = base_dir / "tmp" / "runtime" / "results"
        log.warning(f"Reconstructed results_dir: {results_dir}")
    else:
        # Cannot reconstruct, plotting might fail to save
        log.error("results_dir variable not found. Plot saving may fail.")
        # Define a dummy path to avoid NameError, but saving won't work
        results_dir = Path("./results")  # Or some other fallback

# --- Plotting Configuration ---
# Use the RESULTS_FILE variable defined in cell 2
RESULTS_FILE_PATH = RESULTS_FILE if "RESULTS_FILE" in locals() else None

PLOT_OUTPUT_FILE = results_dir / "runtime_vs_length_plot.png"  # Changed filename

point_alpha = 0.6
point_size = 5
legend_point_size = 50
legend_point_alpha = 1.0  # Use alpha=1 for legend markers

# Set plotting style and font (consider making this configurable)
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]  # Common sans-serif fonts

log.info("--- Starting Result Plotting ---")

# --- Read and Prepare Data ---
if RESULTS_FILE_PATH and RESULTS_FILE_PATH.exists():
    try:
        log.info(f"Loading benchmark results from: {RESULTS_FILE_PATH}")
        df = pl.read_csv(RESULTS_FILE_PATH, separator="\t", infer_schema_length=1000)

        # Filter for valid runs: runtime > 0, length > 0, and no error recorded
        # Check if 'error' column exists and filter based on it being null or empty
        if "error" in df.columns:
            df_valid = df.filter(
                (pl.col("runtime") > 0)
                & (pl.col("length") > 0)
                & (pl.col("error").is_null() | (pl.col("error") == ""))
            )
        else:
            # Fallback if error column somehow wasn't saved
            log.warning(
                "Column 'error' not found in results. Filtering only on runtime and length."
            )
            df_valid = df.filter((pl.col("runtime") > 0) & (pl.col("length") > 0))

        num_valid_points = df_valid.height
        num_total_points = df.height
        log.info(
            f"Found {num_valid_points} valid data points out of {num_total_points} total rows for plotting."
        )

        if num_valid_points == 0:
            log.warning(
                "No valid data points found after filtering. Cannot generate plot."
            )
        else:
            # Separate data by method for plotting
            inertia_data = df_valid.filter(pl.col("method") == "inertia")
            family_data = df_valid.filter(pl.col("method") == "family")
            log.info(
                f"Inertia points: {inertia_data.height}, Family points: {family_data.height}"
            )

            # --- Create Plot ---
            plt.figure(figsize=(8, 5))  # Adjusted size slightly

            # Plot inertia (Blue)
            plt.scatter(
                inertia_data["length"],
                inertia_data["runtime"],
                color="#1E88E5",  # A slightly darker blue
                label="FlatProt - Inertia",  # Legend label
                alpha=point_alpha,
                s=point_size,
                edgecolors="w",  # White edge for better visibility
                linewidth=0.2,
            )

            # Plot family (Green) - alignment-based
            plt.scatter(
                family_data["length"],
                family_data["runtime"],
                color="#004D40",  # A darker green
                label="FlatProt - Family",  # Legend label
                alpha=point_alpha,
                s=point_size,
                edgecolors="w",
                linewidth=0.2,
            )

            # --- Configure Axes and Title ---
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Protein Length (Number of Residues)")
            plt.ylabel("Runtime (s)")
            plt.title("FlatProt Runtime vs Protein Length (Log-Log Scale)")

            # --- Dynamic Ticks based on Data Range ---
            # Combine data to find overall min/max for robust tick generation
            all_lengths = df_valid["length"]
            all_runtimes = df_valid["runtime"]

            min_len = all_lengths.min() if not all_lengths.is_empty() else 10
            max_len = all_lengths.max() if not all_lengths.is_empty() else 10000
            min_time = all_runtimes.min() if not all_runtimes.is_empty() else 0.01
            max_time = all_runtimes.max() if not all_runtimes.is_empty() else 10

            # Generate log-spaced ticks covering the data range
            x_tick_vals = 10 ** np.arange(
                np.floor(np.log10(min_len)), np.ceil(np.log10(max_len)) + 1
            )
            y_tick_vals = 10 ** np.arange(
                np.floor(np.log10(min_time)), np.ceil(np.log10(max_time)) + 1
            )

            # Format ticks for readability (integer for >= 1, scientific for < 1)
            x_tick_labels = [f"{t:.0f}" if t >= 1 else f"{t:.1g}" for t in x_tick_vals]
            y_tick_labels = [f"{t:.0f}" if t >= 1 else f"{t:.2g}" for t in y_tick_vals]

            plt.xticks(x_tick_vals, x_tick_labels)
            plt.yticks(y_tick_vals, y_tick_labels)
            # Set axis limits slightly beyond ticks if needed, or let matplotlib auto-adjust
            # plt.xlim(min(x_tick_vals)*0.9, max(x_tick_vals)*1.1)
            # plt.ylim(min(y_tick_vals)*0.9, max(y_tick_vals)*1.1)

            # --- Add Legend ---
            # Create custom handles for legend with visible markers
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="FlatProt - Inertia",
                    markersize=legend_point_size**0.5,
                    markerfacecolor="#1E88E5",
                    alpha=legend_point_alpha,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="FlatProt - Family",
                    markersize=legend_point_size**0.5,
                    markerfacecolor="#004D40",
                    alpha=legend_point_alpha,
                ),
            ]
            plt.legend(
                handles=legend_handles, loc="upper left", frameon=True, fontsize="small"
            )

            # --- Add Grid ---
            plt.grid(
                True,
                which="major",
                linestyle="-",
                linewidth="0.5",
                color="gray",
                alpha=0.6,
            )
            plt.grid(
                True,
                which="minor",
                linestyle=":",
                linewidth="0.5",
                color="gray",
                alpha=0.3,
            )

            # --- Save and Show Plot ---
            try:
                # Ensure results directory exists before saving
                PLOT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(PLOT_OUTPUT_FILE, dpi=300, bbox_inches="tight")
                log.info(f"Plot saved successfully to: {PLOT_OUTPUT_FILE}")
            except Exception as save_err:
                log.error(
                    f"Failed to save plot to {PLOT_OUTPUT_FILE}: {save_err}",
                    exc_info=True,
                )

            plt.show()  # Display the plot in the notebook interface

    except FileNotFoundError:
        log.error(f"Results file not found: {RESULTS_FILE_PATH}. Cannot generate plot.")
    except ImportError as import_err:
        log.error(
            f"Required plotting library (matplotlib, numpy) not found: {import_err}. Please install it."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred during plotting: {e}", exc_info=True)

elif not RESULTS_FILE_PATH:
    log.error(
        "RESULTS_FILE variable not defined. Cannot determine results file path for plotting."
    )
else:  # File path exists but file does not
    log.error(
        f"Results file does not exist at the specified path: {RESULTS_FILE_PATH}. Run benchmark step first."
    )

log.info("--- Result Plotting Finished ---")
