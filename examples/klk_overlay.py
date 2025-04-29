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
# Essential Imports (Keep remaining imports here)
import collections
import csv
import xml.etree.ElementTree as ET
import zipfile  # Needed later
from typing import Union
from IPython.display import SVG, display

# FlatProt Core Imports
from flatprot.core import logger

# %%
# --- Configuration & Setup ---
print("\n[STEP 1] Setting up script paths and variables...")

# Define script-specific directories and file paths using base paths
tmp_dir = tmp_dir_base / "klk_overlay"  # Specific tmp dir
data_archive = data_dir_base / "KLK.zip"  # Specific input archive

# Create specific temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")


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
# ## Cluster Structures using Foldseek
#
# Use `foldseek easy-cluster` to group similar structures. We will identify
# representative structures for clusters containing more than one member
# and proceed with only these representatives for alignment and projection.

# %%

# Define directories and paths for clustering
cluster_output_prefix = tmp_dir / "klk_cluster"
clustering_tmp_dir = tmp_dir / "clustering_tmp"
clustering_tmp_dir.mkdir(exist_ok=True)

# Convert paths to strings for the command line
structures_dir_str = str(structures_dir)
cluster_output_prefix_str = str(cluster_output_prefix)
clustering_tmp_dir_str = str(clustering_tmp_dir)

# Run foldseek easy-cluster using subprocess
cluster_cmd_list = [
    "foldseek",
    "easy-cluster",
    structures_dir_str,
    cluster_output_prefix_str,
    clustering_tmp_dir_str,
    "--min-seq-id",
    "0.5",
    "--c",
    "0.9",
    "--threads",
    "4",
    "-v",
    "0",
]


# Define helper locally or assume it's available from imported colab_setup
def run_local_cmd(cmd_list, check=True):
    # Simplified local runner if needed
    print(f" Running local command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, check=check, capture_output=True, text=True)


try:
    # Prefer run_cmd from colab_setup if available
    if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
        # Note: Foldseek easy-cluster can print a lot, capture=False might be better
        # Or capture=True but don't print stdout in run_cmd helper
        colab_setup.run_cmd(cluster_cmd_list, capture=False)
    else:
        run_local_cmd(cluster_cmd_list)  # Fallback
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Foldseek clustering failed: {e}", file=sys.stderr)
except Exception as e:
    print(f"[ERROR] Failed to run Foldseek clustering: {e}", file=sys.stderr)


# Parse the cluster results
cluster_file = Path(f"{cluster_output_prefix_str}_cluster.tsv")
clusters = collections.defaultdict(list)
representatives = set()
all_cluster_members = set()

if cluster_file.exists():
    with open(cluster_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                representative, member = row
                # Foldseek output might lack the .cif extension, add it back
                representative_fn = f"{Path(representative).stem}.cif"
                member_fn = f"{Path(member).stem}.cif"
                clusters[representative_fn].append(member_fn)
                representatives.add(representative_fn)
                all_cluster_members.add(member_fn)
            else:
                print(f"Skipping malformed row: {row}")
else:
    print(f"Error: Cluster file not found at {cluster_file}")

# Filter for clusters with more than one member
large_clusters = {rep: members for rep, members in clusters.items() if len(members) > 1}
representative_files = [structures_dir / rep for rep in large_clusters.keys()]

print(f"Found {len(representatives)} total clusters.")
print(f"Found {len(large_clusters)} clusters with size > 1.")
print(f"Proceeding with {len(representative_files)} representative structures.")


# %% [markdown]
# ## Align Representative Structures
#
# Create directories for alignment outputs (matrices and info files).
# Define the path to the alignment database (adjust if necessary).

# %%
matrix_dir = tmp_dir / "npy"
info_dir = tmp_dir / "json"
matrix_dir.mkdir(exist_ok=True)
info_dir.mkdir(exist_ok=True)


# %% [markdown]
# Run `flatprot align` for each representative structure against the database.

# %%
for file in representative_files:
    matrix_path = matrix_dir / f"{file.stem}_matrix.npy"
    info_path = info_dir / f"{file.stem}_info.json"

    # Construct command list
    align_cmd_list = [
        "uv",
        "run",
        "flatprot",
        "align",
        str(file.resolve()),
        str(matrix_path.resolve()),
        str(info_path.resolve()),
        "--target-db-id",
        "3000114",  # SCOP ID for Trypsin-like serine proteases
        "--quiet",
    ]

    try:
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(align_cmd_list)
        else:
            run_local_cmd(align_cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Alignment failed for {file.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to run alignment for {file.name}: {e}", file=sys.stderr)


# %% [markdown]
# ## Generate Projections using Python API (Store Data)
#
# transformations and projection, then store the projected coordinates and
# the renderable Scene object in memory for later processing.

# %%
svg_dir = tmp_dir / "svg"
svg_dir.mkdir(exist_ok=True)

style_config = """
[helix]
color = "#FF7D7D"
opacity = 0.8 # Base opacity for elements

[sheet]
color = "#7D7DFF"
opacity = 0.8 # Base opacity for elements

[coil]
color = "#AAAAAA"
opacity = 0.5 # Base opacity for elements

[connection]
color = "#AAAAAA"
opacity = 0.5
"""

style_file = tmp_dir / "style.toml"
style_file.write_text(style_config)


# Define Canvas size for intermediate rendering (large enough)
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500

# Store cluster counts for opacity calculation
cluster_counts = {rep: len(members) for rep, members in large_clusters.items()}


def calculate_opacity(cluster_counts, min_opacity=0.05, max_opacity=1.0):
    """Calculates opacity for each representative based on its cluster size."""
    opacities = {}
    if not cluster_counts:
        return opacities

    counts = list(cluster_counts.values())
    min_count = min(counts) if counts else 1
    max_count = max(counts) if counts else 1

    for representative, count in cluster_counts.items():
        if max_count == min_count:
            normalized_count = 1.0
        else:
            normalized_count = (count - min_count) / (max_count - min_count)
        opacity = min_opacity + normalized_count * (max_opacity - min_opacity)
        opacities[representative] = opacity
    return opacities


# Calculate opacities for the representatives we are processing
representative_opacities = calculate_opacity(cluster_counts)


# %% [markdown]
# Loop through the representative `.cif` files, apply transformations,

# %%
print(f"Processing {len(representative_files)} structures for projection...")

for file in representative_files:
    matrix_path = str(matrix_dir / f"{file.stem}_matrix.npy")
    file_str = str(file.resolve())
    style_str = str(style_file.resolve())
    svg_path = str(svg_dir / f"{file.stem}.svg")

    rep_base_name = file.stem  # Used as key

    project_cmd_list = [
        "uv",
        "run",
        "flatprot",
        "project",
        file_str,
        "--matrix",
        matrix_path,
        "-o",
        svg_path,
        "--quiet",
        "--canvas-width",
        str(CANVAS_WIDTH),
        "--canvas-height",
        str(CANVAS_HEIGHT),
        "--style",
        style_str,
    ]

    try:
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(project_cmd_list)
        else:
            run_local_cmd(project_cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Projection failed for {file.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to run projection for {file.name}: {e}", file=sys.stderr)

print("Structure processing finished.")


# %% [markdown]
# ## Create Overlay from Stored Data (Fixed Viewbox)
#
# Assemble the final SVG by rendering each stored scene, extracting
# its content, and merging it into a new SVG with scaled opacity and a fixed viewbox.

# %%
# --- Define Fixed Viewbox ---
viewbox_x = 0
viewbox_y = 0
viewbox_width = CANVAS_WIDTH
viewbox_height = CANVAS_HEIGHT
viewbox_str = (
    f"{viewbox_x:.0f} {viewbox_y:.0f} {viewbox_width:.0f} {viewbox_height:.0f}"
)
print(f"Using Fixed ViewBox: {viewbox_str}")

# --- SVG Assembly ---

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
# Registering with a prefix might help make output cleaner if ET uses it
ET.register_namespace("svg", SVG_NAMESPACE)

# Use namespaced element creation
combined_svg_root = ET.Element(
    f"{{{SVG_NAMESPACE}}}svg",
    {
        "xmlns": SVG_NAMESPACE,
        "viewBox": viewbox_str,
        "width": str(int(viewbox_width)),
        "height": str(int(viewbox_height)),
    },
)
combined_defs = ET.SubElement(combined_svg_root, f"{{{SVG_NAMESPACE}}}defs")
unique_defs_ids = set()

print(
    f"Assembling final SVG from {len(representative_files)} representative structures..."
)
processed_count = 0
final_svg_path = None

# Iterate through the generated SVG files
# for svg_file_path in svg_files:
# Iterate through the representative structure files
for representative_file in representative_files:
    rep_base_name = representative_file.stem
    svg_file_path = svg_dir / f"{rep_base_name}.svg"

    # Get the original cluster rep name (without _matrix suffix if present)
    # cluster_rep_key = rep_base_name.replace("_matrix", "")
    # Use rep_base_name directly as the key for opacity
    opacity = representative_opacities.get(representative_file.name, 1.0)

    try:
        # Read the SVG file content
        with open(svg_file_path, "r", encoding="utf-8") as f:
            svg_string = f.read()

        # Parse the SVG content
        parser = ET.XMLParser(encoding="utf-8")
        individual_root = ET.fromstring(svg_string.encode("utf-8"), parser=parser)
        xml_namespaces = {"svg": SVG_NAMESPACE}

        # --- Merge Defs ---
        defs_element = individual_root.find("svg:defs", xml_namespaces)
        if defs_element is not None:
            for elem in list(defs_element):
                elem_id = elem.get("id")
                if elem_id:
                    if elem_id not in unique_defs_ids:
                        combined_defs.append(elem)
                        unique_defs_ids.add(elem_id)
                else:
                    combined_defs.append(elem)

        # --- Create Group and Copy Content ---
        svg_group = ET.SubElement(
            combined_svg_root,
            f"{{{SVG_NAMESPACE}}}g",
            {"opacity": f"{opacity:.3f}", "id": f"group_{rep_base_name}"},
        )

        # Copy graphical content
        for element in individual_root:
            if element.tag == f"{{{SVG_NAMESPACE}}}defs":
                continue
            if (
                element.tag == f"{{{SVG_NAMESPACE}}}rect"
                and element.get("id") == "background"
            ):
                continue
            # IMPORTANT: Assume copied elements *might* not be properly namespaced
            # We rely on the parent group (svg_group) and root (combined_svg_root)
            # having the correct namespace declaration (xmlns) for rendering engines.
            svg_group.append(element)

        processed_count += 1

    except FileNotFoundError:
        print(f"  -> WARNING: SVG file not found: {svg_file_path}")
    except ET.ParseError as pe:
        print(f"  -> WARNING: Could not parse rendered SVG file {svg_file_path}: {pe}")
    except Exception as e:
        print(
            f"  -> WARNING: Error processing SVG file {svg_file_path} during assembly: {e}"
        )
        logger.error(f"Error processing file {svg_file_path}", exc_info=True)

# --- Save Final SVG ---
if processed_count > 0:
    final_svg_path = tmp_dir / "overlay_fixed_viewbox.svg"
    tree = ET.ElementTree(combined_svg_root)
    try:
        # Write to a file handle, *without* default_namespace
        with open(str(final_svg_path), "wb") as f_out:
            tree.write(f_out, encoding="utf-8", xml_declaration=True)
        print(f"Created overlay of {processed_count} structures at {final_svg_path}")
    except IOError as e:
        print(f"Error writing final overlay SVG: {e}")
        final_svg_path = None
else:
    print("No scenes were processed successfully, skipping final SVG generation.")

# %% [markdown]
# Display the final overlay SVG.
# The overlay now shows representative structures aligned to SCOP Superfamily 3000114,
# with a fixed viewbox and opacity scaled by cluster size.

# %%
# Display the overlay SVG in the notebook
if final_svg_path and final_svg_path.exists():
    display(SVG(filename=str(final_svg_path)))
    print(
        "Overlay of aligned representative protein structures (fixed viewbox, scaled opacity):"
    )
else:
    print("Final overlay SVG generation failed or was skipped.")
