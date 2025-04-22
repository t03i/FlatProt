# %% [markdown]
# # FlatProt: Domain-Specific Alignment and Projection with Comparison
#
# **Goal:** Generate and compare different 2D SVG visualizations of a protein structure based on its domains:
# 1.  **Normal Projection:** Standard inertia-based projection of the full structure.
# 2.  **Domain-Aligned Projection:** Domains aligned individually, reassembled, then projected.
# 3.  **Domain-Separated Projection:** Domains identified and spatially separated in the 2D projection.
#
# **Workflow:**
# 1.  Setup: Configure paths and parameters.
# 2.  Load Structure: Load the original PDB/CIF file into a `flatprot.core.Structure`.
# 3.  *(New)* Generate Normal Projection SVG.
# 4.  Load Domains: Parse domain definitions (ResidueRanges) from Chainsaw output.
# 5.  Align Domains: Extract, align (Foldseek), retrieve transformation matrices (`flatprot.utils.domain_utils.DomainTransformation`).
# 6.  Apply Transformations: Apply matrices to the original structure -> `transformed_structure`.
# 7.  Project Transformed: Orthographically project `transformed_structure` -> `projected_transformed_structure`.
# 8.  Generate Domain-Aligned Projection SVG from `projected_transformed_structure`.
# 9.  *(New)* Generate Domain-Separated Projection SVG using the normal projection coordinates and domain definitions for layout.
# 10. Display outputs.
#
# **Note on Structure Types:**
# - `gemmi.Structure`: Used internally by the `extract_domain` helper function for reading/writing CIF files via the Gemmi library.
# - `flatprot.core.Structure`: The primary data structure used throughout the FlatProt workflow for representing structure, coordinates, and chains.

# %% [markdown]
# ---
# ## Step 1: Setup and Imports
#
# Import necessary libraries and configure input/output paths and parameters.

# %%
# Essential Imports
import os
import shutil
import traceback
from pathlib import Path
from typing import List, Optional, Dict

# Third-party Libraries
import gemmi
import polars as pl
import numpy as np

from flatprot.scene.structure.base_structure import BaseStructureStyle

print("[INFO] Imported standard libraries.")

# FlatProt Components
try:
    from flatprot.core import Structure, ResidueRange, FlatProtError
    from flatprot.io import (
        GemmiStructureParser,
        validate_structure_file,
        InvalidStructureError,
        OutputFileError,
    )
    from flatprot.alignment import (
        AlignmentDatabase,
        align_structure_database,
        get_aligned_rotation_database,
        AlignmentResult,
    )
    from flatprot.transformation import TransformationMatrix, TransformationError

    # Import specific transformation functions needed
    from flatprot.utils.structure_utils import (
        project_structure_orthographically,
        transform_structure_with_inertia,
    )
    from flatprot.utils.domain_utils import (
        DomainTransformation,
        apply_domain_transformations_masked,
        create_domain_aware_scene,
    )
    from flatprot.utils.scene_utils import (
        create_scene_from_structure,
    )
    from flatprot.scene import BaseStructureStyle, AreaAnnotationStyle
    from flatprot.renderers import SVGRenderer

    # For displaying SVGs inline in Jupyter
    from IPython.display import display, SVG

    print("[INFO] Successfully imported FlatProt components and IPython display.")
except ImportError as e:
    print(f"[ERROR] Failed to import components: {e}")
    print("        Please ensure FlatProt and IPython are installed correctly.")
    raise SystemExit(1)

# --- User Configuration ---

# Input Structure Details
structure_id: str = "1kt0"  # PDB ID or identifier for file naming
data_base_dir: Path = Path("../data")  # Root directory for data/databases

# Domain Definition File (from Chainsaw)
structure_dir: Path = data_base_dir / structure_id
chainsaw_file: Path = structure_dir / f"{structure_id.lower()}-chainsaw-domains.tsv"

# Primary Structure File (PDB or CIF)
structure_file: Path = structure_dir / f"{structure_id}.cif"

# Temporary Directory for Intermediate Files
tmp_dir: Path = Path("../tmp/domain_alignment_projection")

# Alignment & Foldseek Databases (!!! VERIFY THESE PATHS !!!)
database_base_dir: Path = data_base_dir / "databases"
alignment_db_dir: Path = Path("../out/alignment_db")
foldseek_db_dir: Path = alignment_db_dir / "foldseek"
db_file_path: Path = alignment_db_dir / "alignments.h5"  # Use updated directory
foldseek_db_path: Path = foldseek_db_dir / "db"  # Foldseek search database

# Foldseek Configuration
foldseek_path: str = "foldseek"  # Path to executable (or name if in PATH)
min_probability: float = 0.5  # Minimum probability for accepting an alignment

# Output Configuration (!!! Distinct filenames !!!)
output_svg_normal: Path = tmp_dir / f"{structure_id}-normal-projection.svg"
output_svg_aligned: Path = tmp_dir / f"{structure_id}-domains-aligned-reassembled.svg"
output_svg_separated: Path = tmp_dir / f"{structure_id}-domains-separated-layout.svg"

# Canvas & Layout
canvas_width: int = 1000
canvas_height: int = 1000
domain_separation_spacing: float = 100.0  # Pixels between separated domains
domain_separation_arrangement: str = "horizontal"  # or "vertical"

styles_dict: Optional[Dict[str, BaseStructureStyle]] = {
    "area_annotation": AreaAnnotationStyle(
        fill_color="#ddd",
        fill_opacity=0.4,
        line_style=(4, 2),
        stroke_width=2.0,
        label_offset=(-190, -150),
    ),
}

# --- End Configuration ---

# Setup Environment
os.makedirs(tmp_dir, exist_ok=True)
print(f"[SETUP] Using temporary directory: {tmp_dir.resolve()}")
print("[SETUP] Output Files:")
print(f"         Normal Projection: {output_svg_normal.resolve()}")
print(f"         Domain-Aligned:    {output_svg_aligned.resolve()}")
print(f"         Domain-Separated:  {output_svg_separated.resolve()}")


# Validate Foldseek executable early
if not Path(foldseek_path).exists() and not shutil.which(foldseek_path):
    print(
        f"[ERROR] FoldSeek executable not found at '{foldseek_path}' or in system PATH."
    )
    print("        Please install Foldseek or correct the 'foldseek_path' variable.")
    raise SystemExit(1)
else:
    print(
        f"[SETUP] Foldseek executable found: {shutil.which(foldseek_path) or Path(foldseek_path).resolve()}"
    )

# %% [markdown]
# ---
# ## Helper Function: `extract_domain`
#
# (Uses `gemmi.Structure` internally for file I/O)


# %%
def extract_domain(
    struct_file: Path, chain: str, start_res: int, end_res: int, output_file: Path
) -> None:
    """Extracts a domain from a structure file using Gemmi."""
    # (Implementation unchanged from previous version - see above)
    if not struct_file.exists():
        raise FileNotFoundError(f"Input structure file not found: {struct_file}")
    try:
        structure = gemmi.read_structure(
            str(struct_file), merge_chain_parts=True, format=gemmi.CoorFormat.Detect
        )
        domain = gemmi.Structure()
        safe_output_stem = output_file.stem.replace(":", "_").replace("-", "_")
        domain.name = safe_output_stem
        model = gemmi.Model("1")
        original_chain_instance: Optional[gemmi.Chain] = None
        if structure and len(structure) > 0:
            for ch in structure[0]:
                if ch.name == chain:
                    original_chain_instance = ch
                    break
        if original_chain_instance is None:
            raise ValueError(f"Chain '{chain}' not found in {struct_file}")
        new_chain = gemmi.Chain(chain)
        extracted_residues_count = 0
        for residue in original_chain_instance:
            seq_id = residue.seqid.num
            if start_res <= seq_id <= end_res:
                new_chain.add_residue(residue.clone())
                extracted_residues_count += 1
        if extracted_residues_count == 0:
            raise ValueError(
                f"No residues in range {start_res}-{end_res} for chain '{chain}' in {struct_file}"
            )
        model.add_chain(new_chain)
        domain.add_model(model)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        domain.make_mmcif_document().write_file(str(output_file))
        print(
            f"      [OK] Extracted: chain {chain}, {start_res}-{end_res} ({extracted_residues_count} res) -> {output_file.name}"
        )
    except (ValueError, FileNotFoundError) as ve:
        raise ve
    except Exception as e:
        raise Exception(
            f"Gemmi error processing {struct_file} for {chain}:{start_res}-{end_res}: {e}"
        ) from e


print("[INFO] Helper function `extract_domain` defined.")

# %% [markdown]
# ---
# ## Step 2: Load Full Structure
#
# (Loads into `flatprot.core.Structure`)

# %%
print("\n[STEP 2] Loading Full Structure...")
print(f"         Input file: {structure_file.resolve()}")

original_structure: Optional[Structure] = None
try:
    validate_structure_file(structure_file)
    parser = GemmiStructureParser()
    original_structure = parser.parse_structure(structure_file)
    if original_structure is None:
        raise InvalidStructureError("Structure parsing returned None.")
    coord_count = (
        original_structure.coordinates.shape[0]
        if original_structure.coordinates is not None
        else 0
    )
    print(
        f"[DONE] Structure '{original_structure.id}' loaded ({len(original_structure)} res, {coord_count} coords)."
    )
    if coord_count == 0:
        print("[WARN] Structure has no coordinates.")
except (FileNotFoundError, InvalidStructureError, FlatProtError) as e:
    print(f"[ERROR] Load failed: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected load error: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 3: Generate Normal Projection SVG
#
# Apply standard inertia transformation and orthographic projection to the original structure.

# %%
print("\n[STEP 3] Generating Normal Projection SVG...")
print(f"         Output file: {output_svg_normal.resolve()}")

projected_original_structure: Optional[Structure] = None  # To store the result
try:
    if original_structure is None or original_structure.coordinates is None:
        raise ValueError("Original structure or its coordinates are missing.")

    # 1. Apply Inertia Transformation
    print("         Applying inertia transformation...")
    inertia_transformed_structure = transform_structure_with_inertia(original_structure)
    if inertia_transformed_structure.coordinates is None:
        raise TransformationError("Inertia transform removed coords.")
    print(
        f"         Inertia transformed shape: {inertia_transformed_structure.coordinates.shape}"
    )

    # 2. Project Orthographically
    print(
        f"         Applying orthographic projection ({canvas_width}x{canvas_height})..."
    )
    projected_original_structure = project_structure_orthographically(
        inertia_transformed_structure, canvas_width, canvas_height
    )
    if projected_original_structure.coordinates is None:
        raise FlatProtError("Projection removed coords.")
    print(f"         Projected shape: {projected_original_structure.coordinates.shape}")

    # 3. Create Scene and Render
    print("         Creating scene...")
    # Optional: Load styles if you want them applied to the normal projection too

    scene_normal = create_scene_from_structure(
        projected_original_structure, styles_dict
    )
    # Optional: Add annotations if desired for the normal view
    # if annotations_file:
    #     try:
    #         add_annotations_to_scene(annotations_file, scene_normal)
    #     except Exception as e:
    #          print(f"[WARN] Failed annotations: {e}")

    print("         Rendering SVG...")
    renderer_normal = SVGRenderer(scene_normal, canvas_width, canvas_height)
    renderer_normal.save_svg(output_svg_normal)
    print("[DONE] Normal projection SVG saved.")

except (ValueError, TransformationError, FlatProtError) as e:
    print(f"[ERROR] Failed to generate normal projection: {e}")
    # We might want to continue to other steps if possible, or halt:
    # raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error in normal projection: {e}")
    traceback.print_exc()
    # raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 4: Load Domain Definitions
#
# (Parse `ResidueRange` objects from Chainsaw file)

# %%
print("\n[STEP 4] Loading Domain Definitions...")
print(f"         Chainsaw file: {chainsaw_file.resolve()}")

defined_domains: List[ResidueRange] = []  # List of flatprot.core.ResidueRange
try:
    # (Implementation unchanged - parses TSV to defined_domains list)
    if not chainsaw_file.exists():
        raise FileNotFoundError(f"Chainsaw file not found: {chainsaw_file}")
    domains_df = pl.read_csv(chainsaw_file, separator="\t")
    structure_id_upper = structure_id.upper()
    filtered_df = domains_df.filter(
        pl.col("chain_id").str.to_uppercase() == structure_id_upper
    )
    if len(filtered_df) == 0:
        raise ValueError(f"No entries for ID '{structure_id}' in {chainsaw_file}")
    if len(filtered_df) > 1:
        print(f"[WARN] Multiple entries for ID '{structure_id}'. Using first.")
    chopping_str = filtered_df["chopping"][0]
    print(f"         Parsing chopping string: '{chopping_str}'")
    raw_ranges = chopping_str.split(",")
    for range_str in raw_ranges:
        range_str = range_str.strip()
        chain_id = "A"
        res_range_part = range_str
        if not range_str:
            continue
        if ":" in range_str:
            parts = range_str.split(":", 1)
            if len(parts) == 2 and parts[0].strip():
                chain_id = parts[0].strip().upper()
                res_range_part = parts[1].strip()
            else:
                print(f"[WARN] Skipping malformed range: '{range_str}'")
                continue
        if "-" in res_range_part:
            try:
                start_str, end_str = res_range_part.split("-", 1)
                start_res, end_res = int(start_str.strip()), int(end_str.strip())
                if start_res <= 0 or end_res <= 0 or start_res > end_res:
                    print(f"[WARN] Skipping invalid numbers: {range_str}")
                    continue
                defined_domains.append(
                    ResidueRange(chain_id, start_res, end_res)
                )  # Note: coord_index not needed here
            except ValueError:
                print(f"[WARN] Skipping non-integer range: '{range_str}'")
                continue
        else:
            print(f"[WARN] Skipping range missing '-': '{range_str}'")
            continue
    if not defined_domains:
        raise ValueError(f"No valid ranges parsed from: '{chopping_str}'")
    print(f"[DONE] Loaded {len(defined_domains)} domain definitions:")
    for dom in defined_domains:
        print(f"         - {dom}")
except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as e:
    print(f"[ERROR] Domain load failed: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected domain load error: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 5: Align Domains and Collect Transformations
#
# (Extract, Align via Foldseek, Retrieve `DomainTransformation` objects)

# %%
print(
    f"\n[STEP 5] Aligning Domains & Collecting Transformations ({len(defined_domains)} total)..."
)
print(f"         Alignment DB: {db_file_path.resolve()}")
print(f"         Foldseek DB: {foldseek_db_path.resolve()}")

# --- Pre-checks & Init ---
if not db_file_path.exists():
    print("[ERROR] Alignment DB not found")
    raise SystemExit(1)
if not foldseek_db_path.exists():
    print("[ERROR] Foldseek DB dir not found")
    raise SystemExit(1)
if original_structure is None:
    print("[ERROR] Original structure not loaded.")
    raise SystemExit(1)
try:
    alignment_db = AlignmentDatabase(db_file_path)
    print("[INFO] Alignment DB connection initialized.")
except Exception as e:
    print(f"[ERROR] Failed to init alignment DB: {e}")
    raise SystemExit(1)

domain_transformations: List[DomainTransformation] = []  # Stores results
failed_domains: List[ResidueRange] = []
domain_scop_ids: Dict[str, str] = {}  # Store domain_id -> scop_id mapping

# --- Domain Processing Loop ---
for i, domain_range in enumerate(defined_domains):
    print(f"\n   Processing Domain {i + 1}/{len(defined_domains)}: {domain_range}")
    domain_range_str_safe = (
        f"{domain_range.chain_id}_{domain_range.start}_{domain_range.end}"
    )
    domain_file_name = f"{structure_id}_domain_{domain_range_str_safe}.cif"
    temp_domain_file = tmp_dir / domain_file_name
    alignment_result: Optional[AlignmentResult] = None
    domain_matrix: Optional[TransformationMatrix] = None
    print("      [1/4] Extracting...")
    extract_domain(
        structure_file,
        domain_range.chain_id,
        domain_range.start,
        domain_range.end,
        temp_domain_file,
    )
    # 2. Align
    print("      [2/4] Aligning...")
    alignment_result = align_structure_database(
        temp_domain_file, foldseek_db_path, foldseek_path, min_probability
    )
    print(
        f"            -> Hit: {alignment_result.db_id} (P={alignment_result.probability:.3f})"
    )
    # 3. Get Matrix
    print("      [3/4] Retrieving matrix...")
    matrix_result = get_aligned_rotation_database(alignment_result, alignment_db)
    domain_matrix = matrix_result[0]
    db_entry = matrix_result[1]
    print(f"            -> Matrix for {db_entry.entry_id if db_entry else 'N/A'}")

    # Store the DB entry ID (assuming it's the SCOP ID)
    domain_id_str = f"{domain_range.chain_id}:{domain_range.start}-{domain_range.end}"
    if db_entry and db_entry.entry_id:
        domain_scop_ids[domain_id_str] = db_entry.entry_id
        print(f"            -> Associated DB ID: {db_entry.entry_id}")
    else:
        print(f"            -> DB entry or entry_id not found for {domain_id_str}.")

    # 4. Store
    if domain_matrix:
        domain_tf = DomainTransformation(domain_range, domain_matrix, domain_id_str)
        domain_transformations.append(domain_tf)
        print("      [4/4] Stored transformation.")
    else:
        print("      [WARN] Matrix None. Skipping storage.")

    if temp_domain_file.exists():
        try:
            os.remove(temp_domain_file)
        except OSError as e:
            print(f"      [WARN] Cleanup failed: {e}")

# --- Post-loop Summary ---
print(
    f"\n[DONE] Finished domain alignment. Collected {len(domain_transformations)} transformations."
)
if failed_domains:
    print(f"         Failed domains: {len(failed_domains)}")
if not domain_transformations:
    print("[ERROR] No transformations collected. Cannot proceed.")
    raise SystemExit(1)


# %% [markdown]
# ---
# ## Step 6: Apply Domain Transformations
#
# (Apply collected matrices to the *original* structure -> `transformed_structure`)

# %%
print(f"\n[STEP 6] Applying {len(domain_transformations)} Domain Transformations...")

transformed_structure: Optional[Structure] = None
try:
    if original_structure is None or original_structure.coordinates is None:
        raise ValueError("Original structure/coords missing.")
    print(f"         Original shape: {original_structure.coordinates.shape}")
    transformed_structure = apply_domain_transformations_masked(
        original_structure, domain_transformations
    )
    print("[DONE] Transformations applied.")
    if transformed_structure is None or transformed_structure.coordinates is None:
        raise TransformationError("Transform removed coords.")
    print(f"         Transformed shape: {transformed_structure.coordinates.shape}")
    if transformed_structure.coordinates.shape != original_structure.coordinates.shape:
        print("[WARN] Shape changed!")
except (ValueError, TransformationError, FlatProtError) as e:
    print(f"[ERROR] Apply failed: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected apply error: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 7: Project Transformed Structure to 2D
#
# (Orthographically project the `transformed_structure` from Step 6 -> `projected_transformed_structure`)

# %%
print("\n[STEP 7] Projecting Domain-Transformed Structure...")
print(f"         Target canvas: {canvas_width}x{canvas_height}px")

projected_transformed_structure: Optional[Structure] = None
try:
    if transformed_structure is None or transformed_structure.coordinates is None:
        raise ValueError("Transformed structure/coords missing.")
    print(f"         Input shape (3D): {transformed_structure.coordinates.shape}")
    projected_transformed_structure = project_structure_orthographically(
        transformed_structure, canvas_width, canvas_height
    )
    print("[DONE] Projection completed.")
    if (
        projected_transformed_structure is None
        or projected_transformed_structure.coordinates is None
    ):
        raise FlatProtError("Projection removed coords.")
    print(
        f"         Output shape (2D): {projected_transformed_structure.coordinates.shape}"
    )
    if projected_transformed_structure.coordinates.shape[1] != 2:
        print(
            f"[WARN] Expected 2D coords, got {projected_transformed_structure.coordinates.shape}"
        )
except (ValueError, FlatProtError) as e:
    print(f"[ERROR] Projection failed: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected projection error: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 8: Generate Domain-Aligned Projection SVG
#
# (Render the `projected_transformed_structure` from Step 7)

# %%
print("\n[STEP 8] Rendering Domain-Aligned Projection SVG...")
print(f"         Output file: {output_svg_aligned.resolve()}")

try:
    if projected_transformed_structure is None:
        raise ValueError("Projected transformed structure missing.")

    # --- Create the scene for the aligned structure --- #
    print("         Creating scene for domain-aligned structure...")
    scene_aligned = create_scene_from_structure(
        projected_transformed_structure, styles_dict
    )
    print("         Scene created.")
    # Optional: Add annotations if they apply to the transformed view

    print("         Rendering SVG...")
    renderer_aligned = SVGRenderer(scene_aligned, canvas_width, canvas_height)
    renderer_aligned.save_svg(output_svg_aligned)
    print("[DONE] Domain-aligned SVG saved.")
except (ValueError, FileNotFoundError, OutputFileError, FlatProtError) as e:
    print(f"[ERROR] SVG render failed: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"[ERROR] Unexpected render error: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# %% [markdown]
# ---
# ## Step 9: Generate Domain-Separated Projection SVG
#
# Use the *normal* projection (from Step 3) coordinates but group elements by domain and apply spatial layout.

# %%
print("\n[STEP 9] Generating Domain-Separated Projection SVG...")
print(f"         Output file: {output_svg_separated.resolve()}")
print(
    f"         Layout: {domain_separation_arrangement}, Spacing: {domain_separation_spacing}px"
)

try:
    if projected_original_structure is None:
        raise ValueError(
            "Normal projected structure (from Step 3) is missing. Cannot create separated view."
        )
    if not defined_domains:
        raise ValueError(
            "Domain definitions (from Step 4) are missing. Cannot create separated view."
        )

    # Create DomainTransformation objects for layout (using identity matrices)
    domain_definitions_for_layout: List[DomainTransformation] = []
    for domain_range in defined_domains:
        domain_id_str = (
            f"{domain_range.chain_id}:{domain_range.start}-{domain_range.end}"
        )
        # Use identity matrix as we only need the range for grouping/layout here
        layout_tf = DomainTransformation(
            domain_range=domain_range,
            transformation_matrix=TransformationMatrix(
                np.eye(3), np.zeros(3)
            ),  # Use np.eye(4) for identity
            domain_id=domain_id_str,
        )
        domain_definitions_for_layout.append(layout_tf)
    print(f"         Created {len(domain_definitions_for_layout)} layout definitions.")

    # Create the domain-aware scene using the *normally projected* coordinates
    print("         Creating domain-aware scene...")

    scene_separated = create_domain_aware_scene(
        projected_structure=projected_original_structure,  # Use coords from normal projection
        domain_definitions=domain_definitions_for_layout,  # Use ranges for grouping/layout
        spacing=domain_separation_spacing,
        arrangement=domain_separation_arrangement,
        default_styles=styles_dict,
        domain_scop_ids=domain_scop_ids,  # Pass the SCOP ID mapping
    )
    print("         Scene created with domain groups.")
    # Note: Annotations might be complex to apply correctly to separated views unless defined relative to domains

    # Render the separated scene
    print("         Rendering SVG...")
    renderer_separated = SVGRenderer(
        scene_separated, width=canvas_width + 200, height=canvas_height + 200
    )  # Let renderer calculate canvas size based on layout
    # Adjust width/height if you want to force it, but None often works well for separated views
    renderer_separated.save_svg(output_svg_separated)
    print("[DONE] Domain-separated SVG saved.")

except (ValueError, FileNotFoundError, OutputFileError, FlatProtError) as e:
    print(f"[ERROR] Failed to generate domain-separated SVG: {e}")
    # Decide whether to halt or continue
except Exception as e:
    print(f"[ERROR] Unexpected error in domain-separated SVG generation: {e}")
    traceback.print_exc()

# %% [markdown]
# ---
# ## Step 10: Display Results & Reference
#
# Show the generated SVG files for comparison.

# %%
print("\n[STEP 10] Displaying Results...")

# Display SVGs if files exist
for title, svg_path in [
    ("Normal Projection", output_svg_normal),
    ("Domain-Aligned & Reassembled", output_svg_aligned),
    ("Domain-Separated Layout", output_svg_separated),
]:
    if svg_path.exists():
        print(f"\n--- {title} ---")
        display(SVG(filename=str(svg_path)))
    else:
        print(f"\n--- {title} (File not found: {svg_path.name}) ---")


# Add reference
print("\n" + "=" * 80)
print("Reference for Chainsaw Domain Parsing Method:")
print(
    "Wells et al. (2024) Chainsaw: a rapid method for defining protein domains from structure. Bioinformatics, btae296."
)
print("DOI: https://doi.org/10.1093/bioinformatics/btae296")
print("GitHub: https://github.com/JudeWells/chainsaw")
print("=" * 80)

# %% [markdown]
# ---
# ## End of Notebook
#
# Processing complete. The comparison SVGs are generated and displayed above.
