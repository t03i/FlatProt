# %% [markdown]
# # FlatProt: Protein Family Overlay
#
# **Create publication-quality overlays with automatic clustering and alignment!**
#
# One command handles everything: clustering, alignment, and visualization.

# %% [markdown]
# ## 🔧 Setup

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
import subprocess
import time
from pathlib import Path

# Create output directory
output_dir = Path(f"{tmp_path}overlay/klk/")
output_dir.mkdir(parents=True, exist_ok=True)

# Check if files already exist
existing_files = list(output_dir.glob("*.cif"))
if existing_files:
    print(f"📁 Found {len(existing_files)} existing CIF files, skipping extraction")
else:
    print(f"📦 Extracting protein structures from {data_path}KLK.zip...")
    print("⏳ This may take a few minutes (436 files to extract)...")

    try:
        # Use subprocess with timeout and progress indication
        start_time = time.time()
        result = subprocess.run([
            "unzip", "-j", f"{data_path}KLK.zip",
            "KLK/structures/*.cif", "-d", str(output_dir), "-q"
        ], timeout=300, capture_output=True, text=True)

        elapsed = time.time() - start_time
        if result.returncode == 0:
            print(f"✅ Extraction completed in {elapsed:.1f} seconds")
        else:
            print(f"⚠️ Extraction finished with warnings (return code: {result.returncode})")
            if result.stderr:
                print(f"Errors: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Extraction timed out after 5 minutes")
        print("💡 Try running the command manually or check if the ZIP file is corrupted")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("💡 Trying to use existing files or manual extraction needed")

# Count extracted files and check their validity
klk_files = list(output_dir.glob("*.cif"))
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
# *Note: The first time you run this, it will take a while to download the database.*

# %%
# Create the overlay!
!uv run flatprot overlay "{tmp_path}overlay/klk/*.cif" -o "{tmp_path}overlay/overlay.png" --family 3000114 --style "{tmp_path}overlay/style.toml" --canvas-width 800 --canvas-height 600 --clustering --dpi 150 --quiet

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
