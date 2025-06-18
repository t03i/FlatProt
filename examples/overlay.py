# %% [markdown]
# # FlatProt: Protein Family Overlay
#
# **Create publication-quality overlays with automatic clustering and alignment!**
#
# One command handles everything: clustering, alignment, and visualization.

# %% [markdown]
# ## 🔧 Setup (Colab only)

# %%
# Colab setup with robust path handling
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
# ## 📁 Extract protein structures

# %%
# Extract KLK proteins from archive
!mkdir -p "{tmp_path}overlay/klk"
!unzip -j "{data_path}KLK.zip" "klk/structures/*.cif" -d "{tmp_path}overlay/klk/" 2>/dev/null || echo "Using existing files"

# Count extracted files and check their validity
from pathlib import Path
klk_files = list(Path(f"{tmp_path}overlay/klk/").glob("*.cif"))
print(f"📁 {len(klk_files)} protein structures ready")

# %% [markdown]
# ## 🎨 Create custom style

# %%
# Create custom style file
with open(f"{tmp_path}overlay/style.toml", "w") as f:
    f.write("""[helix]
color = "#FF6B6B"
opacity = 0.8

[sheet]
color = "#4ECDC4"
opacity = 0.8

[coil]
color = "#95A5A6"
opacity = 0.6
""")
print(f"📝 Style file created at {tmp_path}overlay/style.toml")

# %% [markdown]
# ## 🎯 **The Magic Command**
#
# This single command:
# 1. Clusters similar structures
# 2. Aligns to Trypsin family (3000114)
# 3. Creates overlay with smart opacity
# 4. Applies custom styling

# %%
# Create the overlay!
!uv run flatprot overlay "{tmp_path}overlay/klk/*.cif" -o "{tmp_path}overlay/overlay.png" --family 3000114 --style "{tmp_path}overlay/style.toml" --canvas-width 800 --canvas-height 600 --clustering --dpi 150

# %% [markdown]
# ## 🎉 View your overlay

# %%
from IPython.display import Image, display
from pathlib import Path

overlay_file = f"{tmp_path}overlay/overlay.png"
if Path(overlay_file).exists():
    display(Image(overlay_file))
    print("🧬 KLK protein family overlay - multiple structures aligned and overlaid!")
    print("💡 Opacity shows cluster sizes - darker = more similar structures")
else:
    print("❌ Overlay creation failed - check the command above")

# %% [markdown]
# ## 🎓 What happened?
#
# **In one command, FlatProt:**
# - Found similar protein structures automatically
# - Aligned them to a reference family framework
# - Created a overlay visualization
# - Applied your custom color scheme
#
# **What would normally take considerable time and effort!** ✨
