# %% [markdown]
# # FlatProt: Protein Domain Splitting
#
# **Extract and visualize protein domains separately!**
#
# This example shows how to use `flatprot split` to extract structural domains
# and create individual visualizations for comparative analysis.

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
# ## 📖 About the Example
#
# We'll work with **1KT0** - a multi-domain protein with three distinct structural regions:
# - **Domain 1**: [1kt0A01](https://www.cathdb.info/domain/1kt0A01)
# - **Domain 2**: [1kt0A02](https://www.cathdb.info/domain/1kt0A02)
# - **Domain 3**: [1kt0A03](https://www.cathdb.info/domain/1kt0A03)
#
# with domain 1 and 2 being from the same superfamily and domain 3 being related to the binding of the protein.
# The `flatprot split` command extracts these domains and creates individual projections; alignment to the included superfamily database validates that the domain separation is picked up by SCOP as well.

# %% [markdown]
# ## 🔍 Examine the Domain Information

# %%
# Look at the domain definition file
domain_file = f"{data_path}1KT0/1kt0-chainsaw-domains.tsv"
with open(domain_file, 'r') as f:
    lines = f.readlines()

print("🧬 Domain definitions for 1KT0:")
print("=" * 50)
for line in lines:
    print(line.strip())

# %% [markdown]
# ## ✂️ **Parse Domain Regions**
#
# Extract the domain boundaries from the ChainSaw file:

# %%
# Parse domain regions from the TSV file
with open(domain_file, 'r') as f:
    lines = f.readlines()

# Get the domain regions from the second line (skip header)
data_line = lines[1].strip().split('\t')
regions_str = data_line[4]  # The chopping column
chain_id = data_line[0]     # Chain ID

# Convert to flatprot format (add chain prefix)
regions_with_chain = ','.join([f"A:{region}" for region in regions_str.split(',')])

print(f"🧬 Parsed domains: {regions_str}")
print(f"🎯 FlatProt format: {regions_with_chain}")

# %% [markdown]
# ## ✂️ **The Magic of Split**
#
# One command extracts all domains and creates aligned visualization:
# *Note: The first time you run this, it will take a while to download the database.*

# %%
# Create split domain visualization
!mkdir -p "{tmp_path}split"
!uv run flatprot split "{data_path}1KT0/1kt0.cif" --regions "{regions_with_chain}" --output "{tmp_path}split/1kt0_domains.svg" --canvas-width 800 --canvas-height 600 --show-positions minimal --show-database-alignment --gap-x 200 --gap-y 20 --quiet

print("✅ Domain splitting completed!")

# %% [markdown]
# ## 🎉 View the Split Domain Visualization

# %%
from IPython.display import HTML, display
from pathlib import Path

# Display the split domain visualization
print("🧬 1KT0 Protein Domains - Split View:")
print()

split_file = f"{tmp_path}split/1kt0_domains.svg"
if Path(split_file).exists():
    print("🎯 Multi-domain protein with aligned regions:")

    with open(split_file) as f:
        svg_content = f.read()

    # Make SVG responsive
    svg_content = svg_content.replace('<svg ', '<svg style="max-width: 100%; height: auto;" ', 1)
    display(HTML(f'<div style="text-align: center; margin: 20px; border: 2px solid #4ECDC4; padding: 15px; border-radius: 10px; background-color: #f8f9fa;">{svg_content}</div>'))

    print("🔍 **What you see:**")
    print("   • Three domains extracted and aligned")
    print("   • Domains positioned with gaps for clarity")
    print("   • Each domain maintains its structural features")
else:
    print("❌ Split visualization not found - check the split command above")




# %% [markdown]
# ## 🎓 What you accomplished:
#
# 1. **Parsed** domain boundaries from ChainSaw predictions
# 2. **Extracted** multiple structural domains from a single protein
# 3. **Created** aligned split visualization showing all domains
# 4. **Visualized** domain organization and structural relationships

# %% [markdown]
# ## 💡 **The Power of Domain Splitting**
#
# This workflow demonstrates:
# - **Automated domain extraction** from structural data
# - **Individual visualization** of complex multi-domain proteins
# - **Comparative analysis** at the domain level
# - **Simplified interpretation** of protein architecture
#
# **Perfect for structural biology and protein evolution studies!** 🎯✨
