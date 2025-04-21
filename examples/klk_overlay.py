# %% [markdown]
# # FlatProt: KLK Structure Alignment and Overlay Example
#
# This example script demonstrates how to align KLK (Kallikrein) structures and overlay their FlatProt projections.
#
# ## Overview
#
# This script performs the following steps:
#
# 1. Sets up temporary directories.
# 2. Extracts KLK structure files (.cif) from a zip archive.
# 3. Aligns each KLK structure to a common reference frame (SCOP Superfamily 3000114 - Trypsin-like serine proteases) using `flatprot align` and a specified Foldseek database.
# 4. Generates 2D projections (.svg) for each aligned structure using `flatprot project`, applying the alignment transformation matrix.
# 5. Creates a style file for the projections.
# 6. Overlays all generated SVG projections into a single SVG file.
# 7. Displays the final overlay.
#
# It is designed to be run interactively, potentially converted from/to a
# Jupyter Notebook using Jupytext.

# %% [markdown]
# ## Setup
#
# Import necessary libraries and define a helper magic command `pybash`
# to run shell commands within the Python script, substituting Python variables.

# %%
from pathlib import Path
import os
import zipfile
from typing import Union
from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import SVG, display
import drawsvg as draw

ipython = get_ipython()


@register_cell_magic
def pybash(line: str, cell: str) -> None:
    """Execute cell contents as a bash script, substituting Python variables.

    Args:
        line: The arguments passed to the magic command (unused).
        cell: The content of the cell to execute. Python variables can be
              interpolated using f-string-like syntax (e.g., {variable_name}).
    """
    if ipython:
        ipython.run_cell_magic("bash", "", cell.format(**globals()))


# %% [markdown]
# Define paths for temporary data, the input data archive,
# and create the temporary directory.

# %%
tmp_dir = Path("../tmp/klk_overlay")
data_archive = Path("../data/KLK.zip")

os.makedirs(tmp_dir, exist_ok=True)


# %% [markdown]
# ## Extract KLK Structures
#
# Define a function to extract only the `.cif` files from the `KLK/`
# directory within the zip archive.


# %%
def extract_klk_folder(
    archive_path: Union[str, Path], output_dir: Union[str, Path]
) -> None:
    """Extract KLK .cif files from a zip archive to the specified output directory.

    Only files located within the 'KLK/' directory inside the archive and
    ending with '.cif' will be extracted. Files are placed directly into the
    `output_dir`, stripping the 'KLK/' prefix from their paths.

    Args:
        archive_path: Path to the zip archive file.
        output_dir: Directory where the KLK .cif files will be extracted.
    """
    # Convert to Path objects if they're strings
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract only the KLK folder contents directly into output_dir
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        klk_files = [
            f
            for f in zip_ref.namelist()
            if (f.startswith("KLK/") and f.endswith(".cif"))
        ]
        for file in klk_files:
            file_path = Path(file)
            target_path = output_dir / file_path.name
            # Extract the file content
            with zip_ref.open(file) as source, open(target_path, "wb") as target:
                target.write(source.read())

    print(f"Extracted KLK folder contents from {archive_path} to {output_dir}")


# %% [markdown]
# Create the target directory for structures and run the extraction function.

# %%
# Create structures directory
structures_dir = tmp_dir / "structures"
structures_dir.mkdir(exist_ok=True)

# Extract only the KLK folder from the archive to the structures directory
extract_klk_folder(data_archive, structures_dir)

# %% [markdown]
# ## Align Structures
#
# Create directories for alignment outputs (matrices and info files).
# Define the path to the alignment database (adjust if necessary).

# %%
matrix_dir = tmp_dir / "npy"
info_dir = tmp_dir / "json"
matrix_dir.mkdir(exist_ok=True)
info_dir.mkdir(exist_ok=True)

# Adjust this path to your actual Foldseek database location
alignment_db_path = "../out/alignment_db"
min_probability = 0.5  # Example minimum probability threshold

# %% [markdown]
# Run `flatprot align` for each structure against the database.

# %%
for file in structures_dir.glob("*.cif"):
    matrix_path = matrix_dir / f"{file.stem}_matrix.npy"
    info_path = info_dir / f"{file.stem}_info.json"

    # Convert paths to strings for the command line
    file_str = str(file)
    matrix_path_str = str(matrix_path)
    info_path_str = str(info_path)

    ipython.run_cell_magic(
        "pybash",
        "",
        "uv run flatprot align {file_str} {matrix_path_str} {info_path_str} "
        "-d {alignment_db_path} --min-probability {min_probability} "
        "--target-db-id 3000114 --quiet",
    )


# %% [markdown]
# ## Generate Projections
#
# Create a directory for the SVG outputs and define a simple style
# configuration for the FlatProt projections.

# %%
svg_dir = tmp_dir / "svg"
svg_dir.mkdir(exist_ok=True)

style = """
[helix]
color = "#FF7D7D"
opacity = 0.1


[sheet]
color = "#7D7DFF"
opacity = 0.1

[coil]
color = "#777777"
opacity = 0.1

"""

style_file = tmp_dir / "style.toml"
style_file.write_text(style)

# %% [markdown]
# Loop through the extracted `.cif` files, generate an SVG projection for each
# using the `flatprot project` command, **applying the alignment matrix**.
# The alignment ensures all structures are oriented according to the target superfamily (3000114).

# %%
for file in structures_dir.glob("*.cif"):
    matrix_path = matrix_dir / f"{file.stem}_matrix.npy"
    svg_path = svg_dir / f"{file.stem}.svg"

    # Convert paths to strings for the command line
    file_str = str(file)
    matrix_path_str = str(matrix_path)
    svg_path_str = str(svg_path)
    style_file_str = str(style_file)

    # Use pybash magic to run the command, now including the matrix
    ipython.run_cell_magic(
        "pybash",
        "",
        "uv run flatprot project {file_str} {svg_path_str} "
        "--matrix {matrix_path_str} --quiet --style {style_file_str}",
    )


# %% [markdown]
# ## Create and Display Overlay
#
# Use the `drawsvg` library to create an overlay of all generated SVGs.

# %%

# Create a new drawing for the overlay
overlay = draw.Drawing(800, 600)

# Add each SVG to the overlay
for svg_file in svg_dir.glob("*.svg"):
    # Read the SVG content
    svg_content = svg_file.read_text()

    # Extract the SVG elements (skip the first line which is the XML declaration)
    svg_elements = (
        svg_content.split("\n", 1)[1] if "<?xml" in svg_content else svg_content
    )

    # Add the SVG content as a group to the overlay
    # We use a foreignObject to embed the SVG content directly
    group = draw.Raw(svg_elements)
    overlay.append(group)

# Save the overlay SVG
overlay_path = tmp_dir / "overlay.svg"
overlay.save_svg(str(overlay_path))

print(
    f"Created overlay of {len(list(svg_dir.glob('*.svg')))} SVG files at {overlay_path}"
)

# %% [markdown]
# Display the final overlay SVG.
# The overlay now shows structures aligned to SCOP Superfamily 3000114.

# %%
# Display the overlay SVG in the notebook

# Load and display the overlay SVG
display(SVG(str(overlay_path)))

# You can also add a title before displaying the SVG
print("Overlay of aligned protein structures:")
