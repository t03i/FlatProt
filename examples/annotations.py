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

# --- Path Definitions --- (This section remains largely the same)
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
# Essential Imports
from pathlib import Path  # Keep this one
import os  # Keep this one

# IPython Specifics for Bash Magic and Display
try:
    from IPython.display import SVG, display
except ImportError:
    print("[WARN] IPython not found. SVG display will not work.", file=sys.stderr)
    # Define dummy functions if IPython is not available
    SVG = lambda x: print(f"Cannot display SVG: {x}")
    display = lambda x: print(f"Cannot display: {x}")

# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define script-specific directories and file paths using base paths
tmp_dir = tmp_dir_base / "files_example"  # Specific tmp dir
structure_file = data_dir_base / "1KT0" / "1kt0.cif"  # Specific input file

structure_svg = tmp_dir / "1kt0_styled_annotated.svg"  # More descriptive name
style_file = tmp_dir / "custom_style.toml"
annotations_file = tmp_dir / "custom_annotations.toml"

# Ensure input structure file exists (after potential download)
if not structure_file.exists():
    print(f"[ERROR] Input structure file not found: {structure_file}")
    if IN_COLAB:
        print(
            "      Check if 'data/1KT0/1kt0.cif' exists in the repository or was downloaded correctly."
        )
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


# Define helper locally or assume it's available from imported colab_setup
def run_local_cmd(cmd_list, check=True):
    # Simplified local runner if needed
    print(f" Running local command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, check=check, capture_output=True, text=True)


# Check if config files were written successfully before proceeding
if style_file.exists() and annotations_file.exists():
    # Construct the command arguments as a list for subprocess
    project_cmd_list = [
        "uv",
        "run",
        "flatprot",
        "project",
        str(structure_file.resolve()),
        str(structure_svg.resolve()),
        "--style",
        str(style_file.resolve()),
        "--annotations",
        str(annotations_file.resolve()),
        "--quiet",
    ]

    # Run the command using subprocess
    print(f"  Running command: {' '.join(project_cmd_list)}")
    try:
        # Prefer run_cmd from colab_setup if available
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(project_cmd_list)
        else:  # Fallback if run_cmd isn't defined (e.g., running outside Colab context)
            run_local_cmd(project_cmd_list)

        print(f"  SVG projection saved to: {structure_svg.resolve()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] flatprot project command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}", file=sys.stderr)
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
