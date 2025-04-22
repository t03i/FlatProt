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
# ## Step 1: Setup and Imports
#
# Import necessary libraries and define file paths for input structures and output results.

# %%
# Essential Imports
from pathlib import Path
import os
from typing import List, Optional

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

# Define base directories
base_dir = Path("..")
data_dir = base_dir / "data" / "3Ftx"
tmp_dir = base_dir / "tmp" / "3ftx_alignment"
db_dir = base_dir / "out"  # Assuming db is here

# Input structure files
cobra_file = data_dir / "cobra.cif"
krait_file = data_dir / "krait.cif"
snake_file = data_dir / "snake.cif"

# Ensure data directory exists
if not data_dir.exists():
    print(f"[ERROR] Data directory not found: {data_dir}")
    # Handle error appropriately, e.g., raise FileNotFoundError or exit
    raise FileNotFoundError(f"Data directory not found: {data_dir}")


# Create temporary directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
print(f"[INFO] Using temporary directory: {tmp_dir.resolve()}")

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

# Alignment database path
db_path = str((db_dir / "alignment_db").resolve())  # Ensure path is correct

# Alignment parameter
min_p = 0.5

print("[INFO] Paths configured:")
print(f"  Input Cobra: {cobra_path}")
print(f"  Input Krait: {krait_path}")
print(f"  Input Snake: {snake_path}")
print(f"  Output Dir: {tmp_dir.resolve()}")
print(f"  Database Path: {db_path}")
print(f"  Min Probability: {min_p}")

# %% [markdown]
# ---
# ## Step 2: Align Structures
#
# Run `flatprot align` for each toxin structure. This command searches the specified database (`-d {db_path}`) for the best alignment above a minimum probability (`--min-probability {min_p}`). It saves the transformation matrix (`{cobra_matrix}`, etc.) and alignment information (`{cobra_info}`, etc.).

# %%
print("\n[STEP 2] Running FlatProt Alignments...")
if ipython:  # Ensure we are in an IPython environment
    # Align Cobra
    print("Aligning Cobra...")
    cobra_align_cmd = f"uv run flatprot align {cobra_path} {cobra_matrix} {cobra_info} -d {db_path} --min-probability {min_p} --quiet"
    ipython.run_cell_magic("pybash", "", cobra_align_cmd)

    # Align Krait
    print("Aligning Krait...")
    krait_align_cmd = f"uv run flatprot align {krait_path} {krait_matrix} {krait_info} -d {db_path} --min-probability {min_p} --quiet"
    ipython.run_cell_magic("pybash", "", krait_align_cmd)

    # Align Snake
    print("Aligning Snake...")
    snake_align_cmd = f"uv run flatprot align {snake_path} {snake_matrix} {snake_info} -d {db_path} --min-probability {min_p} --quiet"
    ipython.run_cell_magic("pybash", "", snake_align_cmd)
else:
    print("[WARN] Not in IPython. Skipping alignment commands.")

print("[INFO] Alignments complete. Matrices and info files generated.")

# %% [markdown]
# ---
# ## Step 3: Project Structures
#
# Run `flatprot project` for each toxin. This command takes the original structure file (`{cobra_path}`, etc.) and the transformation matrix generated in the previous step (`--matrix {cobra_matrix}`, etc.) to create a 2D projection saved as an SVG file (`-o {cobra_out}`, etc.).

# %%
print("\n[STEP 3] Running FlatProt Projections...")
if ipython:  # Ensure we are in an IPython environment
    # Project Cobra
    canvas_args = "--canvas-width 300 --canvas-height 200"
    print("Projecting Cobra...")
    cobra_project_cmd = f"uv run flatprot project {cobra_path} -o {cobra_out} --matrix {cobra_matrix} --quiet {canvas_args}"
    ipython.run_cell_magic("pybash", "", cobra_project_cmd)

    # Project Krait
    print("Projecting Krait...")
    krait_project_cmd = f"uv run flatprot project {krait_path} -o {krait_out} --matrix {krait_matrix} --quiet {canvas_args}"
    ipython.run_cell_magic("pybash", "", krait_project_cmd)

    # Project Snake
    print("Projecting Snake...")
    snake_project_cmd = f"uv run flatprot project {snake_path} -o {snake_out} --matrix {snake_matrix} --quiet {canvas_args}"
    ipython.run_cell_magic("pybash", "", snake_project_cmd)
else:
    print("[WARN] Not in IPython. Skipping projection commands.")


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
