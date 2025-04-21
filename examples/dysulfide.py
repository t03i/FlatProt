# %% [markdown]
# # FlatProt: Cystine Bridge Annotation Example (Cobra Toxin)
#
# **Goal:** This notebook demonstrates how to:
# 1. Compute cystine (disulfide) bridges from a protein structure file (`.cif`).
# 2. Create a FlatProt annotation file (`.toml`) highlighting these bridges.
# 3. Generate a 2D SVG projection of the protein using `flatprot project`, applying the annotations.
#
# **Workflow:**
# 1.  **Setup:** Define paths and import libraries.
# 2.  **Compute Bridges:** Define and use a function (`compute_cystine_bridges`) to identify S-S bonds based on atom distances in the input structure.
# 3.  **Create Annotations:** Define and use a function (`create_cystine_bridge_annotations`) to generate a TOML file describing the identified bridges as line annotations.
# 4.  **Project with Annotations:** Run `flatprot project`, providing the structure file and the generated annotations file.
# 5.  **Display Result:** Show the resulting SVG with the annotated disulfide bonds.

# %% [markdown]
# ---
# ## Step 1: Setup and Imports
#
# Import necessary libraries and define file paths. Setup the `pybash` magic command for executing shell commands if running in an IPython environment.

# %%
# Essential Imports
from pathlib import Path
import os
import toml
import numpy as np
import gemmi
from typing import List, Tuple

# IPython Specifics for Bash Magic and Display
from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import display, HTML

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
tmp_dir = base_dir / "tmp" / "cobra_annotation"
data_dir = base_dir / "data" / "3Ftx"

cobra_file = data_dir / "cobra.cif"
cobra_annotation = tmp_dir / "cobra_annotation.toml"
cobra_svg = tmp_dir / "cobra.svg"

# Ensure data directory exists
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Create temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")

print("[INFO] Paths configured:")
print(f"  Input Cobra CIF: {cobra_file.resolve()}")
print(f"  Output Annotation TOML: {cobra_annotation.resolve()}")
print(f"  Output SVG: {cobra_svg.resolve()}")

# %% [markdown]
# ---
# ## Step 2: Define Function to Compute Cystine Bridges
#
# This function uses the `gemmi` library to parse the structure file, find cysteine residues, and identify pairs whose sulfur atoms (SG) are within a defined distance threshold, indicating a disulfide bond.


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

    print(f"  Found {len(cysteines)} cysteine residues.")

    # Identify disulfide bonds based on distance between sulfur atoms
    # Typical S-S bond distance is around 2.05 Å
    disulfide_threshold = 2.3  # Angstroms, allowing for some flexibility
    bridges: List[Tuple[int, str, int, str]] = []
    found_pairs = set()

    for i in range(len(cysteines)):
        for j in range(i + 1, len(cysteines)):
            res_i, chain_i, pos_i = cysteines[i]
            res_j, chain_j, pos_j = cysteines[j]

            # Calculate distance between sulfur atoms
            distance = np.linalg.norm(pos_i - pos_j)

            if distance <= disulfide_threshold:
                # Store pairs canonically (lower index first) to avoid duplicates if symmetric
                pair = tuple(sorted([(res_i, chain_i), (res_j, chain_j)]))
                if pair not in found_pairs:
                    bridges.append((res_i, chain_i, res_j, chain_j))
                    found_pairs.add(pair)
                    print(
                        f"    Found bridge: {chain_i}:{res_i} <-> {chain_j}:{res_j} (Dist: {distance:.2f} Å)"
                    )

    if not bridges:
        print("  No disulfide bridges found within the threshold.")

    return bridges


# %% [markdown]
# ---
# ## Step 3: Define Function to Create Annotation File
#
# This function takes the list of identified bridges and formats them into a TOML file suitable for FlatProt's annotation system. Each bridge is represented as a line connecting the two cysteine residues.


# %%
def create_cystine_bridge_annotations(
    bridges: List[Tuple[int, str, int, str]], output_path: Path
) -> None:
    """Create a TOML annotation file for cystine bridges.

    Formats a list of cystine bridges into a TOML file for FlatProt visualization.

    Args:
        bridges: List of tuples representing bridges:
                 (residue_index_1, chain_id_1, residue_index_2, chain_id_2).
        output_path: Path where the TOML file should be saved.
    """
    print(f"  Creating annotation file: {output_path.name}")
    annotations = []
    for i, (res1, chain1, res2, chain2) in enumerate(bridges, 1):
        annotation = {
            "label": f"SS_Bridge_{i}",  # Use a consistent prefix
            "type": "line",
            "indices": [
                f"{chain1}:{res1}",
                f"{chain2}:{res2}",
            ],  # Format: CHAIN:RESID
            # Optional styling (add more as needed)
            "style": {
                "stroke": "#FFD700",  # Gold color for visibility
                "stroke_width": 1.5,
                "stroke_dasharray": "4 2",  # Dashed line
            },
        }
        annotations.append(annotation)

    toml_content = {"annotations": annotations}

    # Write the TOML file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(toml_content, f)
        print(
            f"    Successfully wrote {len(annotations)} annotations to {output_path.resolve()}"
        )
    except IOError as e:
        print(f"[ERROR] Failed to write annotation file {output_path}: {e}")


# %% [markdown]
# ---
# ## Step 4: Compute Bridges and Create Annotations
#
# Run the defined functions to find the bridges in the cobra structure and generate the corresponding TOML annotation file.

# %%
print("\n[STEP 4] Computing bridges and creating annotations...")
try:
    cystine_bridges = compute_cystine_bridges(cobra_file)
    print(f"Found {len(cystine_bridges)} cystine bridge(s).")

    if cystine_bridges:
        create_cystine_bridge_annotations(cystine_bridges, cobra_annotation)
    else:
        print("Skipping annotation file creation as no bridges were found.")
except (FileNotFoundError, Exception) as e:
    print(f"[ERROR] Failed during bridge computation or annotation: {e}")
    # Optionally, raise SystemExit or handle differently


# %% [markdown]
# ---
# ## Step 5: Generate Projection with Annotations
#
# Use `flatprot project` to create the SVG visualization. The `--annotations` flag points to the TOML file generated in the previous step, which will draw lines representing the disulfide bonds on the projection.

# %%
print("\n[STEP 5] Generating FlatProt projection with annotations...")

# Check if annotation file was created successfully before proceeding
if cobra_annotation.exists() and ipython:
    # Construct the command
    project_cmd = f"uv run flatprot project {cobra_file} -o {cobra_svg} --annotations {cobra_annotation} --quiet"

    # Run the command using pybash magic via run_cell_magic
    print(f"  Running command: {project_cmd}")
    try:
        ipython.run_cell_magic("pybash", "", project_cmd)
        print(f"  SVG projection saved to: {cobra_svg.resolve()}")
    except Exception as e:
        print(f"[ERROR] flatprot project command failed: {e}")

elif not cobra_annotation.exists():
    print("[WARN] Annotation file does not exist. Skipping projection.")
elif not ipython:
    print("[WARN] Not in IPython environment. Skipping projection command.")

# %% [markdown]
# ---
# ## Step 6: Display Result
#
# Load and display the generated SVG file. If the process was successful, you should see the cobra toxin structure with lines (likely gold and dashed, based on the style defined) connecting the cysteine residues involved in disulfide bonds.

# %%
print("\n[STEP 6] Displaying the final SVG...")

if cobra_svg.exists():
    try:
        with open(cobra_svg, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Modify SVG for better display (responsive width)
        svg_content = svg_content.replace(
            "<svg ",
            '<svg style="width: 80%; height: auto; display: block; margin: auto;" ',
            1,  # Replace only the first occurrence
        )

        html = f"""
        <div style="border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 8px; background-color: #f8f8f8;">
            <h3 style="text-align: center; margin-bottom: 10px;">Cobra Toxin with Cystine Bridges</h3>
            {svg_content}
        </div>
        """
        display(HTML(html))
    except Exception as e:
        print(f"[ERROR] Failed to read or display SVG {cobra_svg}: {e}")
else:
    print(f"[INFO] SVG file not found at {cobra_svg.resolve()}. Cannot display.")

print("\n[INFO] Notebook execution finished.")

# %% [markdown]
# ---
# End of Notebook
# ---
