# %% [markdown]
# # FlatProt: Style and Annotation Example
#
# **Goal:** This notebook demonstrates how to apply custom styles and various annotations (point, line, area) to a FlatProt projection.
#
# **Workflow:**
# 1.  **Setup:** Define paths, import libraries, and setup the `pybash` magic.
# 2.  **Define Annotations:** Create a string containing TOML definitions for point, line, and area annotations.
# 3.  **Define Styles:** Create a string containing TOML style definitions for secondary structure elements (helix, sheet, coil).
# 4.  **Write Files:** Save the annotation and style strings to their respective `.toml` files.
# 5.  **Generate Projection:** Run `flatprot project` using the input structure, specifying the created style and annotation files.
# 6.  **Display Result:** Show the generated SVG, which should reflect the custom styles and annotations.

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
             (COLAB_BASE_DIR / "out").mkdir(exist_ok=True)
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
# out_dir not needed for this specific script

# Ensure base tmp directory exists
tmp_dir_base.mkdir(parents=True, exist_ok=True)


# %%
# Essential Imports
from pathlib import Path
import os

# IPython Specifics for Bash Magic and Display
from IPython.display import SVG, display


# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define script-specific directories and file paths using base paths
tmp_dir = tmp_dir_base / "files_example" # Specific tmp dir
structure_file = data_dir_base / "1KT0" / "1kt0.cif" # Specific input file

structure_svg = tmp_dir / "1kt0_styled_annotated.svg"  # More descriptive name
style_file = tmp_dir / "custom_style.toml"
annotations_file = tmp_dir / "custom_annotations.toml"

# Ensure input structure file exists (after potential download)
if not structure_file.exists():
    print(f"[ERROR] Input structure file not found: {structure_file}")
    if IN_COLAB:
         print("      Check if 'data/1KT0/1kt0.cif' exists in the repository or was downloaded correctly.")
    raise FileNotFoundError(f"Input structure file not found: {structure_file}")

# Create specific temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")

print("[INFO] Paths configured:")
print(f"  Input Structure: {structure_file.resolve()}")
print(f"  Output SVG: {structure_svg.resolve()}")
print(f"  Style File: {style_file.resolve()}")
print(f"  Annotations File: {annotations_file.resolve()}")

# %% [markdown]
# ---
# ## Step 2: Define Annotation Content
#
# Define the content for the annotations file as a multiline string in TOML format. This includes examples of point, line, and area annotations with optional styling.

# %%
print("\n[STEP 2] Defining annotation content...")
annotations_content = """
# FlatProt Annotation Example
# Demonstrates point, line, and area annotations with styles.

[[annotations]]
type = "point"
label = "Active Site His"
index = "A:67" # Example index, verify if correct for 1kt0
[annotations.style]
marker_shape = "triangle" # Example shape
marker_radius = 2
color = "#FF0000" # Red
label_offset = [5,-5]

[[annotations]]
type = "point"
label = "Binding Pocket Tyr"
index = "A:151" # Example index
[annotations.style]
marker_shape = "circle"
marker_radius = 2
color = "#0000FF" # Blue

[[annotations]]
type = "line"
label = "Interaction"
indices = ["A:67", "A:57"] # Example indices for interaction
[annotations.style]
line_color = "#FF8C00" # Dark Orange
stroke_width = 2
line_style = [5,3]
label_color = "#FF8C00"
label_font_size = 10

[[annotations]]
type = "line"
label = "Domain Linker"
indices = ["A:188", "A:190"] # Example short line
[annotations.style]
line_color = "#8A2BE2" # Blue Violet
stroke_width = 1.5
label_font_size = 9

[[annotations]]
type = "area"
label = "Catalytic Loop"
range = "A:189-195" # Example range
[annotations.style]
fill_color = "#32CD32" # Lime Green
opacity = 0.25
color = "#228B22" # Forest Green
stroke_width = 1

"""
print("[INFO] Annotation content defined.")

# %% [markdown]
# ---
# ## Step 3: Define Style Content
#
# Define the content for the style file as a multiline string in TOML format. This specifies custom colors and opacities for helices, sheets, and coils.

# %%
print("\n[STEP 3] Defining style content...")
style_content = """
# Custom Style definitions for protein secondary structures

[helix]
color = "#E41A1C"   # A distinct red from Colorbrewer Set1
stroke_color = "#A01012" # Darker version
stroke_width = 1
opacity = 0.75

[sheet]
color = "#377EB8"   # A distinct blue from Colorbrewer Set1
stroke_color = "#205588" # Darker version
stroke_width = 1
opacity = 0.75

[coil]
stroke_color = "#888888" # Medium gray
stroke_width = 0.75
opacity = 0.9
"""
print("[INFO] Style content defined.")

# %% [markdown]
# ---
# ## Step 4: Write Configuration Files
#
# Write the defined annotation and style strings to their respective `.toml` files.

# %%
print("\n[STEP 4] Writing configuration files...")
try:
    with style_file.open("w", encoding="utf-8") as f:
        f.write(style_content)
    print(f"  Successfully wrote style file: {style_file.resolve()}")
except IOError as e:
    print(f"[ERROR] Failed to write style file {style_file}: {e}")

try:
    with annotations_file.open("w", encoding="utf-8") as f:
        f.write(annotations_content)
    print(f"  Successfully wrote annotations file: {annotations_file.resolve()}")
except IOError as e:
    print(f"[ERROR] Failed to write annotations file {annotations_file}: {e}")

# %% [markdown]
# ---
# ## Step 5: Generate Projection with Styles and Annotations
#
# Run the `flatprot project` command, providing the structure file, the output SVG path, and paths to the newly created style and annotation files.

# %%
print("\n[STEP 5] Generating FlatProt projection with styles and annotations...")

# Check if config files were written successfully before proceeding
if style_file.exists() and annotations_file.exists():
    # Construct the command
    project_cmd = (
        f"uv run flatprot project {structure_file.resolve()} "
        f"{structure_svg.resolve()} "
        f"--style {style_file.resolve()} "
        f"--annotations {annotations_file.resolve()} "
        f"--quiet"
    )

    # Run the command using ! magic
    print("  Running command: flatprot project ...")  # Keep it concise
    try:
        !{project_cmd}
        print(f"  SVG projection saved to: {structure_svg.resolve()}")
    except Exception as e:
        print(f"[ERROR] flatprot project command failed: {e}")

elif not style_file.exists() or not annotations_file.exists():
    print("[WARN] Style or annotations file does not exist. Skipping projection.")

# %% [markdown]
# ---
# ## Step 6: Display Result
#
# Load and display the generated SVG file. Observe the custom colors applied to secondary structures and the various annotations overlaid on the projection.

# %%
print("\n[STEP 6] Displaying the final SVG...")

if structure_svg.exists():
    try:
        # Load and display the SVG
        display(SVG(str(structure_svg)))
        print(f"  Displayed SVG: {structure_svg.name}")
    except Exception as e:
        print(f"[ERROR] Failed to load or display SVG {structure_svg}: {e}")
else:
    print(f"[INFO] SVG file not found at {structure_svg.resolve()}. Cannot display.")

print("\n[INFO] Notebook execution finished.")

# %% [markdown]
# ---
# End of Notebook
# ---
