# %% [markdown]
# # FlatProt Alignment and Projection: Three-Finger Toxins (3FTx)
#
# **Goal:** This notebook demonstrates aligning three related three-finger toxin structures (Cobra, Krait, Snake) using a Foldseek database and then projecting them into 2D SVG visualizations using FlatProt.
#
# **Workflow:**
# 1.  **Setup:** Define paths for input files (CIF) and output directories/files (matrices, info files, SVGs).
# 2.  **Alignment:** Run `flatprot align` for each structure against a pre-computed database to get alignment information and transformation matrices.
# 3.  **Projection:** Run `flatprot project` for each structure, using the corresponding matrix from the alignment step, to generate 2D SVG representations.
# 4.  **Display:** Show the generated SVG files side-by-side for comparison.

# %% [markdown]
# ---
# ## Environment Setup for Google Colab
#
# The following cell checks if the notebook is running in Google Colab and installs the necessary dependencies and downloads required data:
#
# 1.  **FlatProt:** Installs the latest version directly from the GitHub repository using `pip`.
# 2.  **Foldseek:** Downloads (`wget`) and extracts (`tar`) the Foldseek binary (for Linux AVX2) and adds it to the system `PATH`.
# 3.  **DSSP:** Installs the `dssp` package (which provides `mkdssp`) using `apt`.
# 4.  **Repository Data:** Downloads the repository archive, extracts it, and moves the `data/` and `out/` directories to the Colab environment's root.
#
# This setup ensures that the example can run successfully in a Colab environment. If not running in Colab, it assumes dependencies and relative data paths are already correct.

# %%
import os
import sys
from pathlib import Path # Ensure Path is imported here

IN_COLAB = 'google.colab' in sys.modules
COLAB_BASE_DIR = Path(".") # Base directory for Colab CWD (/content)
REPO_DIR_NAME = "FlatProt-main" # Default dir name after unzip

if IN_COLAB:
    print("Running in Google Colab. Setting up environment and data...")

    # --- 1. Install FlatProt ---
    print("\n[1/4] Installing FlatProt...")
    !{sys.executable} -m pip install --quiet --upgrade git+https://github.com/t03i/FlatProt.git#egg=flatprot
    print("FlatProt installation attempted.")

    # --- 2. Install Foldseek ---
    print("\n[2/4] Installing Foldseek...")
    foldseek_url = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
    foldseek_tar = "foldseek-linux-avx2.tar.gz"
    foldseek_dir = "foldseek"
    print(f"Downloading Foldseek from {foldseek_url}...")
    !wget -q {foldseek_url} -O {foldseek_tar}
    print("Extracting Foldseek...")
    !tar -xzf {foldseek_tar}
    foldseek_bin_path = os.path.join(os.getcwd(), foldseek_dir, "bin")
    os.environ['PATH'] = f"{foldseek_bin_path}:{os.environ['PATH']}"
    print(f"Added {foldseek_bin_path} to PATH")
    print("Verifying Foldseek installation...")
    !foldseek --help | head -n 5
    print("Foldseek installation attempted.")

    # --- 3. Install DSSP ---
    print("\n[3/4] Installing DSSP...")
    print("Updating apt package list...")
    !sudo apt-get update -qq
    print("Installing DSSP...")
    !sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dssp
    print("Verifying DSSP installation...")
    !mkdssp --version
    print("DSSP installation attempted.")

    # --- 4. Download Repository Data ---
    print("\n[4/4] Downloading repository data (data/ and out/)...")
    repo_zip_url = "https://github.com/t03i/FlatProt/archive/refs/heads/main.zip"
    repo_zip_file = "repo.zip"
    repo_temp_dir = "repo_temp"

    print(f"Downloading repository archive from {repo_zip_url}...")
    !wget -q {repo_zip_url} -O {repo_zip_file}
    print(f"Extracting archive to {repo_temp_dir}...")
    !unzip -o -q {repo_zip_file} -d {repo_temp_dir}

    extracted_repo_path = COLAB_BASE_DIR / repo_temp_dir / REPO_DIR_NAME
    if extracted_repo_path.is_dir():
         print(f"Moving data/ and out/ directories from {extracted_repo_path}...")
         source_data_path = extracted_repo_path / "data"
         if source_data_path.exists():
             !mv -T {source_data_path} {COLAB_BASE_DIR}/data
             print("Moved data/ directory.")
         else:
             print("[WARN] data/ directory not found in archive.")

         source_out_path = extracted_repo_path / "out"
         if source_out_path.exists():
             !mv -T {source_out_path} {COLAB_BASE_DIR}/out
             print("Moved out/ directory.")
         else:
             print("[INFO] out/ directory not found in archive, creating.")
             (COLAB_BASE_DIR / "out").mkdir(exist_ok=True) # Ensure out exists even if not in archive
    else:
         print(f"[ERROR] Expected directory '{extracted_repo_path}' not found after extraction.")

    print("Cleaning up downloaded files...")
    !rm -rf {repo_temp_dir} {repo_zip_file}

    print("\nEnvironment and data setup complete.")
    base_dir = COLAB_BASE_DIR

# --- Path Definitions ---
print(f"[INFO] Using base directory: {base_dir.resolve()}")
data_dir_base = base_dir / "data"
tmp_dir_base = base_dir / "tmp"
out_dir = base_dir / "out"

# Ensure base tmp/out directories exist
tmp_dir_base.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True) # Ensures out/ exists for db path


# %%
# Essential Imports (keep remaining imports here)
from typing import List, Optional
# IPython Specifics for Bash Magic and Display
from IPython.display import display, HTML


# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define script-specific directories using the base paths
data_dir = data_dir_base / "3Ftx" # Specific data dir for this script
tmp_dir = tmp_dir_base / "3ftx_alignment" # Specific tmp dir
db_dir = out_dir  # Alignment DB expected in out/

# Create specific temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")


# Input structure files (relative to specific data_dir)
cobra_file = data_dir / "cobra.cif"
krait_file = data_dir / "krait.cif"
snake_file = data_dir / "snake.cif"

# Ensure data directory exists (after potential download)
if not data_dir.exists():
    # If we are here, it means base_dir/data/3Ftx doesn't exist
    print(f"[ERROR] Specific data directory not found: {data_dir}")
    if IN_COLAB:
        print("      This might indicate an issue with the repository structure or download.")
    raise FileNotFoundError(f"Data directory not found: {data_dir}")


# Define output file paths within the temporary directory
cobra_path = str(cobra_file.resolve())
cobra_matrix = str(tmp_dir / "cobra_matrix.npy")
cobra_info = str(tmp_dir / "cobra_info.json")
cobra_out = str(tmp_dir / "cobra.svg")

krait_path = str(krait_file.resolve())
krait_matrix = str(tmp_dir / "krait_matrix.npy")
krait_info = str(tmp_dir / "krait_info.json")
krait_out = str(tmp_dir / "krait.svg")

snake_path = str(snake_file.resolve())
snake_matrix = str(tmp_dir / "snake_matrix.npy")
snake_info = str(tmp_dir / "snake_info.json")
snake_out = str(tmp_dir / "snake.svg")

# Alignment parameter
min_p = 0.5

print("[INFO] Paths configured:")
print(f"  Input Cobra: {cobra_path}")
print(f"  Input Krait: {krait_path}")
print(f"  Input Snake: {snake_path}")
print(f"  Output Dir: {tmp_dir.resolve()}")
print(f"  Min Probability: {min_p}")

# %% [markdown]
# ---
# ## Step 2: Align Structures
#
# Run `flatprot align` for each toxin structure. This command searches the specified database (`-d {db_path}`) for the best alignment above a minimum probability (`--min-probability {min_p}`). It saves the transformation matrix (`{cobra_matrix}`, etc.) and alignment information (`{cobra_info}`, etc.).

# %%
print("\n[STEP 2] Running FlatProt Alignments...")
# Remove if ipython check
# Align Cobra
print("Aligning Cobra...")
cobra_align_cmd = f"uv run flatprot align {cobra_path} {cobra_matrix} {cobra_info}  --min-probability {min_p} --quiet"
!{cobra_align_cmd}

# Align Krait
print("Aligning Krait...")
krait_align_cmd = f"uv run flatprot align {krait_path} {krait_matrix} {krait_info} --min-probability {min_p} --quiet"
!{krait_align_cmd}

# Align Snake
print("Aligning Snake...")
snake_align_cmd = f"uv run flatprot align {snake_path} {snake_matrix} {snake_info}  --min-probability {min_p} --quiet"
!{snake_align_cmd}

print("[INFO] Alignments complete. Matrices and info files generated.")

# %% [markdown]
# ---
# ## Step 3: Project Structures
#
# Run `flatprot project` for each toxin. This command takes the original structure file (`{cobra_path}`, etc.) and the transformation matrix generated in the previous step (`--matrix {cobra_matrix}`, etc.) to create a 2D projection saved as an SVG file (`-o {cobra_out}`, etc.).

# %%
print("\n[STEP 3] Running FlatProt Projections...")
# Remove if ipython check
# Project Cobra
canvas_args = "--canvas-width 300 --canvas-height 200"
print("Projecting Cobra...")
cobra_project_cmd = f"uv run flatprot project {cobra_path} -o {cobra_out} --matrix {cobra_matrix} --quiet {canvas_args}"
!{cobra_project_cmd}

# Project Krait
print("Projecting Krait...")
krait_project_cmd = f"uv run flatprot project {krait_path} -o {krait_out} --matrix {krait_matrix} --quiet {canvas_args}"
!{krait_project_cmd}

# Project Snake
print("Projecting Snake...")
snake_project_cmd = f"uv run flatprot project {snake_path} -o {snake_out} --matrix {snake_matrix} --quiet {canvas_args}"
!{snake_project_cmd}

print("[INFO] Projections complete. SVG files generated.")

# %% [markdown]
# ---
# ## Step 4: Display Results
#
# Define a helper function to display the generated SVG files side-by-side within the notebook for easy comparison.

# %%


def display_svg_files(
    svg_files: List[str | Path],
    titles: Optional[List[str]] = None,
    width: str = "30%",
) -> None:
    """
    Display multiple SVG files side by side in a Jupyter environment.

    Args:
        svg_files: A list of paths (as strings or Path objects) to the SVG files.
        titles: An optional list of titles for each SVG. If None, generic titles
                will be used.
        width: The CSS width property for each SVG container (e.g., '30%', '200px').
               Defaults to '30%'.
    """
    if titles is None:
        titles = [f"SVG {i + 1}" for i in range(len(svg_files))]
    elif len(titles) != len(svg_files):
        print(
            "[WARN] Number of titles does not match number of SVG files. Using defaults."
        )
        titles = [f"SVG {i + 1}" for i in range(len(svg_files))]

    html = '<div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap;">'

    for i, (svg_file_path, title) in enumerate(zip(svg_files, titles)):
        svg_path = Path(svg_file_path)  # Ensure it's a Path object
        if not svg_path.exists():
            print(f"[WARN] SVG file not found: {svg_path}. Skipping.")
            html += f"""
            <div style="width: {width}; border: 1px solid #ccc; text-align: center; padding: 10px; margin: 5px; border-radius: 8px; background-color: #f8f8f8;">
                <h3>{title}</h3>
                <p style="color: red;">File not found</p>
            </div>
            """
            continue

        try:
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()

            # Modify SVG to constrain its width and height automatically
            # Ensure responsiveness
            svg_content = svg_content.replace(
                "<svg ",
                '<svg style="width: 100%; height: auto; display: block; margin: auto;" ',
                1,  # Replace only the first occurrence
            )

            html += f"""
            <div style="width: {width}; border: 1px solid #ccc; text-align: center; padding: 10px; margin: 5px; border-radius: 8px; background-color: #f8f8f8;">
                <h3 style="margin-bottom: 10px;">{title}</h3>
                {svg_content}
            </div>
            """
        except Exception as e:
            print(f"[ERROR] Failed to read or process SVG {svg_path}: {e}")
            html += f"""
             <div style="width: {width}; border: 1px solid #ccc; text-align: center; padding: 10px; margin: 5px; border-radius: 8px; background-color: #f8f8f8;">
                 <h3>{title}</h3>
                 <p style="color: red;">Error loading SVG</p>
             </div>
             """

    html += "</div>"
    display(HTML(html))


# %% [markdown]
# Display the three toxin structures side by side using the helper function.

# %%
print("[STEP 4] Displaying Generated SVGs...")
display_svg_files(
    svg_files=[cobra_out, krait_out, snake_out],
    titles=["Cobra Toxin", "Krait Toxin", "Snake Toxin"],
    width="32%",  # Adjust width slightly for better spacing
)

print("[INFO] Notebook execution finished.")

# %% [markdown]
# ---
# End of Notebook
# ---
