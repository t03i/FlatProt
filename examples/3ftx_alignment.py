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
# The following cell checks if the notebook is running in Google Colab
# and runs the shared setup script (`colab_setup.py`) to install
# dependencies and download data.

# %%
import sys
from pathlib import Path
import os  # Keep os import if needed elsewhere
import subprocess  # Keep subprocess import

# Check if in Colab and run setup
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
    # Optionally create it: data_dir_base.mkdir(exist_ok=True)

tmp_dir_base = base_dir / "tmp"

# Ensure base tmp directory exists
tmp_dir_base.mkdir(parents=True, exist_ok=True)


# %%
# Essential Imports (keep remaining imports here)
from typing import List, Optional

# IPython Specifics for Bash Magic and Display
try:
    from IPython.display import display, HTML
except ImportError:
    print("[WARN] IPython not found. SVG display will not work.", file=sys.stderr)
    display, HTML = (
        lambda x: print(f"Cannot display: {x}"),
        lambda x: print(f"Cannot display HTML: {x}"),
    )


# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define script-specific directories using the base paths
data_dir = data_dir_base / "3Ftx"  # Specific data dir for this script
tmp_dir = tmp_dir_base / "3ftx_alignment"  # Specific tmp dir

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
        print(
            "      This might indicate an issue with the repository structure or download."
        )
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


# Define helper locally or assume it's available from imported colab_setup
def run_local_cmd(cmd_list, check=True):
    # Simplified local runner if needed
    print(f" Running local command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, check=check, capture_output=True, text=True)


def run_flatprot_align(input_path, matrix_path, info_path, min_prob):
    cmd = [
        "uv",
        "run",
        "flatprot",
        "align",
        input_path,
        matrix_path,
        info_path,
        "--min-probability",
        str(min_prob),
        "--quiet",
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        # Prefer run_cmd from colab_setup if available
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(cmd)
        else:
            run_local_cmd(cmd)  # Fallback
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Alignment failed for {input_path} with code {e.returncode}",
            file=sys.stderr,
        )
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Alignment failed for {input_path}: {e}", file=sys.stderr)


# Align Cobra
print("Aligning Cobra...")
run_flatprot_align(cobra_path, cobra_matrix, cobra_info, min_p)

# Align Krait
print("Aligning Krait...")
run_flatprot_align(krait_path, krait_matrix, krait_info, min_p)

# Align Snake
print("Aligning Snake...")
run_flatprot_align(snake_path, snake_matrix, snake_info, min_p)

print("[INFO] Alignments complete. Matrices and info files generated.")

# %% [markdown]
# ---
# ## Step 3: Project Structures
#
# Run `flatprot project` for each toxin. This command takes the original structure file (`{cobra_path}`, etc.) and the transformation matrix generated in the previous step (`--matrix {cobra_matrix}`, etc.) to create a 2D projection saved as an SVG file (`-o {cobra_out}`, etc.).

# %%
print("\n[STEP 3] Running FlatProt Projections...")


def run_flatprot_project(input_path, output_path, matrix_path, canvas_args):
    cmd = [
        "uv",
        "run",
        "flatprot",
        "project",
        input_path,
        "-o",
        output_path,
        "--matrix",
        matrix_path,
        "--quiet",
    ]
    # Parse canvas args like "--canvas-width 300 --canvas-height 200"
    cmd.extend(canvas_args.split())
    print(f"  Running: {' '.join(cmd)}")
    try:
        # Prefer run_cmd from colab_setup if available
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(cmd)
        else:
            run_local_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Projection failed for {input_path} with code {e.returncode}",
            file=sys.stderr,
        )
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Projection failed for {input_path}: {e}", file=sys.stderr)


canvas_args = "--canvas-width 300 --canvas-height 200"

# Project Cobra
print("Projecting Cobra...")
run_flatprot_project(cobra_path, cobra_out, cobra_matrix, canvas_args)

# Project Krait
print("Projecting Krait...")
run_flatprot_project(krait_path, krait_out, krait_matrix, canvas_args)

# Project Snake
print("Projecting Snake...")
run_flatprot_project(snake_path, snake_out, snake_matrix, canvas_args)

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
