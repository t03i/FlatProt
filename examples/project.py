# %% [markdown]
# # FlatProt: Simple Protein Projection
#
# **Create 2D protein visualizations with a single command!**
#
# This notebook shows the essence of FlatProt - everything else is just setup for Colab compatibility.

# %% [markdown]
# ## 🔧 Setup

# %%
# Quick Colab setup - skip if running locally
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
# ## 🎯 **This is FlatProt!**
#
# The core command is simple:
# (Note: `flatprot project` creates SVG files - perfect for Jupyter!)

# %%
# Create a protein projection
!mkdir -p "{tmp_path}simple_projection"
!uv run flatprot project "{data_path}3Ftx/cobra.cif" -o "{tmp_path}simple_projection/cobra.svg" --canvas-width 500 --canvas-height 400 --show-positions major


# %% [markdown]
# ## 🎉 View the result

# %%
from IPython.display import Image, HTML
from pathlib import Path


with open(f"{tmp_path}simple_projection/cobra.svg") as f:
    svg = f.read()
svg = svg.replace('<svg ', '<svg style="max-width: 100%; height: auto;" ', 1)
display(HTML(f'<div style="text-align: center;">{svg}</div>'))

print("🧬 Your cobra toxin projection!")

# %% [markdown]
# ## 🎓 What you see:
#
# - 🔴 **Red** = Alpha helices
# - 🔵 **Blue** = Beta sheets
# - ⚪ **Gray** = Loops and coils
# - **Numbers** = Residue positions
#
# **That's it!** FlatProt makes protein visualization simple. 🎯
#
# **Note:** FlatProt generates crisp SVG graphics that scale perfectly in Jupyter notebooks. For PNG output (e.g., for presentations), use `flatprot overlay` instead!
