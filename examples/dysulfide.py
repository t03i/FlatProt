# %% [markdown]
# # FlatProt: Multiple 3FTx Cystine Bridge Annotation Example
#
# **Goal:** This notebook demonstrates how to:
# 1. Find all protein structure files (`.cif`) in a specified directory (`data/3Ftx`).
# 2. For each structure, compute cystine (disulfide) bridges.
# 3. Create a FlatProt annotation file (`.toml`) for each structure, highlighting these bridges as unlabeled, dashed, lime-green lines.
# 4. Generate a 2D SVG projection for each protein using `flatprot project`, applying its specific annotations.
#
# **Workflow:**
# 1.  **Setup:** Define paths and import libraries.
# 2.  **Define Bridge Computation:** Use a function (`compute_cystine_bridges`) to identify S-S bonds based on atom distances.
# 3.  **Define Annotation Creation:** Use a function (`create_cystine_bridge_annotations`) to generate a TOML file describing the identified bridges.
# 4.  **Process Structures:**
#     *   Discover all `.cif` files in the target directory.
#     *   Loop through each file:
#         *   Compute its cystine bridges.
#         *   Create its specific annotation TOML file.
#         *   Run `flatprot project` with the structure and its annotations to generate an SVG.
# 5.  **Summarize Results:** Report the locations of the generated SVG files.

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
out_dir = base_dir / "out"

# Ensure base tmp/out directories exist
tmp_dir_base.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)


# %%
# Essential Imports (Keep remaining imports here)
import toml
import numpy as np
import gemmi # Already checked if not IN_COLAB
from typing import List, Tuple
# IPython Specifics for Bash Magic and Display
from IPython.display import display, HTML


# %%
# --- Configuration ---

print("[STEP 1] Setting up paths and variables...")

# Define script-specific directories using base paths
tmp_dir = tmp_dir_base / "3ftx_dysulfide_annotations" # Specific tmp dir
data_dir = data_dir_base / "3Ftx" # Specific data dir
db_dir = out_dir # Alignment DB expected in out/

# Create specific temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")


# Alignment parameter
min_p = 0.5


print(f"[INFO] Using data directory: {data_dir.resolve()}")
print(f"[INFO] Minimum alignment probability: {min_p}")

# %% [markdown]
# ---
# ## Step 2: Define Function to Compute Cystine Bridges
#
# This function uses the `gemmi` library to parse a structure file, find cysteine residues, and identify pairs whose sulfur atoms (SG) are within a defined distance threshold, indicating a disulfide bond. (No changes needed from the original version).


# %%
def compute_cystine_bridges(structure_path: Path) -> List[Tuple[int, str, int, str]]:
    """
    Compute cystine bridges from a protein structure file.

    Identifies disulfide bonds between cysteine residues by analyzing the
    distance between their sulfur (SG) atoms.

    Args:
        structure_path: Path to the structure file (e.g., CIF format).

    Returns:
        List of tuples, where each tuple represents a bridge and contains
        (residue_index_1, chain_id_1, residue_index_2, chain_id_2).

    Raises:
        FileNotFoundError: If the structure file does not exist.
        Exception: For errors during gemmi parsing.
    """
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    print(f"  Parsing structure: {structure_path.name}")
    try:
        # Load the structure using gemmi
        structure = gemmi.read_structure(
            str(structure_path), merge_chain_parts=True, format=gemmi.CoorFormat.Detect
        )
    except Exception as e:
        raise Exception(f"Error reading structure file {structure_path}: {e}") from e

    # Extract all cysteine residues with their SG atom positions
    cysteines = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == "CYS":
                    # Find the sulfur atom (SG) in each cysteine
                    for atom in residue:
                        if atom.name == "SG":
                            try:
                                res_id_num = int(residue.seqid.num)
                                chain_id_str = str(chain.name)
                                atom_pos = np.array(atom.pos.tolist())
                                cysteines.append((res_id_num, chain_id_str, atom_pos))
                            except (ValueError, AttributeError) as e:
                                print(
                                    f"[WARN] Skipping residue due to parsing error: {residue}, {e}"
                                )
                            break  # Found SG, move to next residue

    # print(f"  Found {len(cysteines)} cysteine residues.") # Less verbose

    # Identify disulfide bonds based on distance between sulfur atoms
    disulfide_threshold = 2.3  # Angstroms
    bridges: List[Tuple[int, str, int, str]] = []
    found_pairs = set()

    for i in range(len(cysteines)):
        for j in range(i + 1, len(cysteines)):
            res_i, chain_i, pos_i = cysteines[i]
            res_j, chain_j, pos_j = cysteines[j]

            distance = np.linalg.norm(pos_i - pos_j)

            if distance <= disulfide_threshold:
                pair = tuple(sorted([(res_i, chain_i), (res_j, chain_j)]))
                if pair not in found_pairs:
                    bridges.append((res_i, chain_i, res_j, chain_j))
                    found_pairs.add(pair)
                    # print(f"    Found bridge: {chain_i}:{res_i} <-> {chain_j}:{res_j} (Dist: {distance:.2f} Å)") # Less verbose

    # if not bridges:
    #     print("  No disulfide bridges found.") # Less verbose

    return bridges


# %% [markdown]
# ---
# ## Step 3: Define Function to Create Annotation File
#
# This function takes the list of identified bridges and formats them into a TOML file suitable for FlatProt's annotation system. Each bridge is represented as an unlabeled, dashed, lime-green line connecting the two cysteine residues.


# %%
def create_cystine_bridge_annotations(
    bridges: List[Tuple[int, str, int, str]], output_path: Path
) -> None:
    """Create a TOML annotation file for cystine bridges.

    Formats a list of cystine bridges into a TOML file for FlatProt visualization,
    styling them as unlabeled, dashed, lime green lines.

    Args:
        bridges: List of tuples representing bridges:
                 (residue_index_1, chain_id_1, residue_index_2, chain_id_2).
        output_path: Path where the TOML file should be saved.
    """
    # print(f"  Creating annotation file: {output_path.name}") # Less verbose
    annotations = []
    for res1, chain1, res2, chain2 in bridges:
        annotation = {
            # "label": f"SS_Bridge_{i}", # Removed label
            "type": "line",
            "indices": [
                f"{chain1}:{res1}",
                f"{chain2}:{res2}",
            ],  # Format: CHAIN:RESID
            "style": {
                "line_color": "#32CD32",  # Lime green color
                "stroke_width": 1.5,
                "line_style": (4, 2),  # Dashed line
                "connector_radius": 0.4,
            },
        }
        annotations.append(annotation)

    toml_content = {"annotations": annotations}

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(toml_content, f)
        # print(f"    Successfully wrote {len(annotations)} annotations to {output_path.resolve()}") # Less verbose
    except IOError as e:
        print(f"[ERROR] Failed to write annotation file {output_path}: {e}")


# %% [markdown]
# ---
# ## Step 4: Process All Structures in the Directory
#
# Discover all `.cif` files in the `data/3Ftx` directory. For each file:
# 1. Compute its disulfide bridges.
# 2. Create the corresponding annotation file.
# 3. Align the structure against the reference database using `flatprot align` to get a transformation matrix.
# 4. Generate the aligned SVG projection using `flatprot project` with the annotation file and the transformation matrix.

# %%
print(f"\n[STEP 4] Processing structures in {data_dir}...")

structure_files = list(data_dir.glob("*.cif"))
if not structure_files:
    print(f"[WARN] No *.cif files found in {data_dir}. Exiting.")
else:
    print(f"Found {len(structure_files)} structure files to process.")

generated_svgs = []

for structure_file in structure_files:
    print(f"\nProcessing: {structure_file.name}")
    file_stem = structure_file.stem
    output_annotation = tmp_dir / f"{file_stem}_annotation.toml"
    output_matrix = tmp_dir / f"{file_stem}_matrix.npy"  # Path for alignment matrix
    output_info = tmp_dir / f"{file_stem}_info.json"  # Path for alignment info
    output_svg = tmp_dir / f"{file_stem}.svg"

    try:
        # 1. Compute Bridges
        print("  Computing bridges...")
        cystine_bridges = compute_cystine_bridges(structure_file)
        print(f"    Found {len(cystine_bridges)} cystine bridge(s).")

        if not cystine_bridges:
            print(
                "  Skipping annotation, alignment, and projection as no bridges were found."
            )
            continue

        # 2. Create Annotations
        print("  Creating annotations...")
        create_cystine_bridge_annotations(cystine_bridges, output_annotation)
        print(f"    Annotation file created: {output_annotation.name}")

        # 3. Align Structure
        print("  Running alignment...")
        align_cmd = "uv run flatprot align {structure_file} {output_matrix} {output_info} --min-probability {min_p} --quiet"
        f_locals_align = {
            "structure_file": structure_file.resolve(),
            "output_matrix": output_matrix.resolve(),
            "output_info": output_info.resolve(),
            "min_p": min_p,
        }
        formatted_align_cmd = align_cmd.format(**f_locals_align)
        alignment_successful = False # Initialize alignment status
        try:
            !{formatted_align_cmd}
            if output_matrix.exists() and output_info.exists():
                print(f"    Alignment likely successful for {structure_file.name}.")
                alignment_successful = True
            else:
                print(
                    f"    [WARN] Alignment command ran, but output files not found for {structure_file.name}."
                )
        except Exception as e:
            print(f"    [ERROR] Alignment failed for {structure_file.name}: {e}")

        # 4. Generate Projection (only if annotations exist and alignment was successful)
        if output_annotation.exists() and alignment_successful:
            print("  Running projection...")
            # Include --matrix argument
            project_cmd = "uv run flatprot project {structure_file} -o {output_svg} --annotations {output_annotation} --matrix {output_matrix} --quiet --canvas-width 300 --canvas-height 500"

            f_locals_project = {
                "structure_file": structure_file.resolve(),
                "output_svg": output_svg.resolve(),
                "output_annotation": output_annotation.resolve(),
                "output_matrix": output_matrix.resolve(),
            }
            formatted_project_cmd = project_cmd.format(**f_locals_project)
            try:
                !{formatted_project_cmd}
                if output_svg.exists():
                    print(f"    SVG projection saved: {output_svg.name}")
                    generated_svgs.append(output_svg)
                else:
                    print(
                        f"    [WARN] Projection command ran, but output SVG not found for {structure_file.name}."
                    )
            except Exception as e:
                print(f"    [ERROR] Projection failed for {structure_file.name}: {e}")

        elif not output_annotation.exists():
            print(
                "  [WARN] Annotation file missing (should not happen if bridges were found). Skipping projection."
            )
        elif not alignment_successful:
            print("  [WARN] Alignment failed or skipped. Skipping projection.")

    except FileNotFoundError as e:
        print(f"  [ERROR] Failed processing {structure_file.name}: {e}")
    except (
        Exception
    ) as e:  # Catch other potential errors during bridge computation/annotation
        print(
            f"  [ERROR] An unexpected error occurred processing {structure_file.name}: {e}"
        )


# %% [markdown]
# ---
# ## Step 5: Summary of Generated SVGs
#
# List the paths of the SVG files generated during the process. You can view these files individually.

# %%
print("\n[STEP 5] Generated SVG Files:")

if generated_svgs:
    for svg_path in generated_svgs:
        print(f"  - {svg_path.resolve()}")
else:
    print("  No SVG files were generated successfully.")

# Example of how to display one SVG if needed (e.g., the first one)
if generated_svgs:
    print("\nDisplaying the first generated SVG as an example:")
    first_svg_path = generated_svgs[0]
    try:
        with open(first_svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        # Basic responsive styling
        svg_content = svg_content.replace(
            "<svg ",
            '<svg style="width: 80%; height: auto; display: block; margin: auto;" ',
            1,
        )
        html = f"""
        <div style="border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 8px; background-color: #f8f8f8;">
            <h3 style="text-align: center; margin-bottom: 10px;">Example: {first_svg_path.name}</h3>
            {svg_content}
        </div>
        """
        display(HTML(html))
    except Exception as e:
        print(f"[ERROR] Failed to read or display SVG {first_svg_path.resolve()}: {e}")


print("\n[INFO] Notebook execution finished.")


# %% [markdown]
# ---
# End of Notebook
# ---
