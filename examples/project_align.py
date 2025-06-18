# %% [markdown]
# # FlatProt: Three-Finger Toxin Alignment
#
# **Compare related protein structures with consistent alignment!**
#
# Shows how to align three similar toxins and create side-by-side projections.

# %% [markdown]
# ## 🔧 Setup

# %%
import sys
from pathlib import Path

def setup_paths():
    """Setup correct paths for both Colab and local environments."""
    if "google.colab" in sys.modules:
        # Colab: Stay in /content/, data will be at /content/data/
        return "data/", "tmp/"
    else:
        # Local: Check if we're in examples/ directory and adjust
        current_dir = Path.cwd()
        if current_dir.name == "examples":
            # We're in examples/, need to go up one level for data access
            project_root = current_dir.parent
            return str(project_root / "data") + "/", str(project_root / "tmp") + "/"
        else:
            # Already in project root
            return "data/", "tmp/"

if "google.colab" in sys.modules:
    !wget -q https://raw.githubusercontent.com/t03i/FlatProt/main/examples/colab_setup.py
    import colab_setup
    colab_setup.setup_colab_environment()

# Get correct paths for this environment
data_path, tmp_path = setup_paths()
print(f"📁 Data path: {data_path}")
print(f"📁 Output path: {tmp_path}")

# %% [markdown]
# ## 🎯 **Step 1: Align structures to reference database**
#
# This finds the best structural alignment for each toxin:
# *Note: The first time you run this, it will take a while to download the database.*

# %%
# Align each structure to reference database
!mkdir -p "{tmp_path}3ftx_align"
!uv run flatprot align "{data_path}3Ftx/cobra.cif" "{tmp_path}3ftx_align/cobra_matrix.npy" "{tmp_path}3ftx_align/cobra_info.json" --min-probability 0.5 --quiet
!uv run flatprot align "{data_path}3Ftx/krait.cif" "{tmp_path}3ftx_align/krait_matrix.npy" "{tmp_path}3ftx_align/krait_info.json" --min-probability 0.5 --quiet
!uv run flatprot align "{data_path}3Ftx/snake.cif" "{tmp_path}3ftx_align/snake_matrix.npy" "{tmp_path}3ftx_align/snake_info.json" --min-probability 0.5 --quiet

print("✅ All structures aligned!")

# %% [markdown]
# ## 🎯 **Step 2: Create aligned projections**
#
# Apply the alignment matrices to ensure consistent orientation:

# %%
# Create projections using alignment matrices
!uv run flatprot project "{data_path}3Ftx/cobra.cif" -o "{tmp_path}3ftx_align/cobra.svg" --matrix "{tmp_path}3ftx_align/cobra_matrix.npy" --canvas-width 400 --canvas-height 300 --show-positions major --quiet
!uv run flatprot project "{data_path}3Ftx/krait.cif" -o "{tmp_path}3ftx_align/krait.svg" --matrix "{tmp_path}3ftx_align/krait_matrix.npy" --canvas-width 400 --canvas-height 300 --show-positions major --quiet
!uv run flatprot project "{data_path}3Ftx/snake.cif" -o "{tmp_path}3ftx_align/snake.svg" --matrix "{tmp_path}3ftx_align/snake_matrix.npy" --canvas-width 400 --canvas-height 300 --show-positions major --quiet

print("✅ All projections created!")

# %% [markdown]
# ## 🎉 Compare the aligned structures

# %%
from IPython.display import Image, HTML, display
from pathlib import Path

# Display the SVG projections
print("🧬 Three-Finger Toxin Family Comparison:")
print()

structure_names = ["cobra", "krait", "snake"]
for name in structure_names:
    svg_file = f"{tmp_path}3ftx_align/{name}.svg"
    if Path(svg_file).exists():
        print(f"🐍 {name.title()} Toxin:")
        with open(svg_file) as f:
            svg = f.read()
        svg = svg.replace('<svg ', '<svg style="max-width: 100%; height: auto;" ', 1)
        display(HTML(f'<div style="text-align: center; margin: 20px; border: 1px solid #ddd; padding: 10px;">{svg}</div>'))
        print()
    else:
        print(f"❌ Could not create {name} projection")

print("\n✨ Notice how all three toxins have the same orientation!")
print("🎯 This makes it easy to compare structural differences and similarities.")

# %% [markdown]
# ## 🎓 What you accomplished:
#
# 1. **Aligned** three related structures to a reference database
# 2. **Generated** transformation matrices for consistent orientation
# 3. **Created** comparable 2D projections
# 4. **Visualized** the conserved three-finger fold
#
# **Key insight:** Despite sequence differences, all three toxins share the same structural framework -
# the characteristic "three fingers" formed by beta-sheet loops! 🧬
#
# This alignment approach works for any protein family in FlatProt's database.
