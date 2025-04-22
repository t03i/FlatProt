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
# ## Step 1: Setup and Imports
#
# Import necessary libraries, define file paths, and setup the `pybash` magic command.

# %%
# Essential Imports
from pathlib import Path
import os

# IPython Specifics for Bash Magic and Display
from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import SVG, display

# %%
# Register pybash magic command if running in IPython
ipython = get_ipython()
if ipython:

    @register_cell_magic
    def pybash(line, cell):
        """Execute bash commands within IPython, substituting Python variables."""
        ipython.run_cell_magic("bash", "", cell.format(**globals()))

else:
    print("[WARN] Not running in IPython environment. `pybash` magic will not work.")

# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define base directories and file paths
base_dir = Path("..")
tmp_dir = base_dir / "tmp" / "files_example"
structure_file = base_dir / "data" / "1KT0" / "1kt0.cif"

structure_svg = tmp_dir / "1kt0_styled_annotated.svg"  # More descriptive name
style_file = tmp_dir / "custom_style.toml"
annotations_file = tmp_dir / "custom_annotations.toml"

# Ensure input structure file exists
if not structure_file.exists():
    raise FileNotFoundError(f"Input structure file not found: {structure_file}")

# Create temporary directory if it doesn't exist
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
if style_file.exists() and annotations_file.exists() and ipython:
    # Construct the command
    project_cmd = (
        f"uv run flatprot project {structure_file.resolve()} "
        f"{structure_svg.resolve()} "
        f"--style {style_file.resolve()} "
        f"--annotations {annotations_file.resolve()} "
        f"--quiet"
    )

    # Run the command using pybash magic via run_cell_magic
    print("  Running command: flatprot project ...")  # Keep it concise
    try:
        ipython.run_cell_magic("pybash", "", project_cmd)
        print(f"  SVG projection saved to: {structure_svg.resolve()}")
    except Exception as e:
        print(f"[ERROR] flatprot project command failed: {e}")

elif not style_file.exists() or not annotations_file.exists():
    print("[WARN] Style or annotations file does not exist. Skipping projection.")
elif not ipython:
    print("[WARN] Not in IPython environment. Skipping projection command.")

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
