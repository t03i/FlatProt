# %% [markdown]
# # FlatProt: Automated Disulfide Bond Detection
#
# **Automatically detect and visualize disulfide bonds across protein families!**
#
# This example shows how to:
# 1. Analyze multiple protein structures automatically
# 2. Find disulfide bonds using structural geometry
# 3. Generate annotation files programmatically
# 4. Create publication-quality visualizations

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

# %%
# Imports for structure analysis
import numpy as np
import gemmi

print("✅ Ready to analyze disulfide bonds!")

# %% [markdown]
# ## 🧬 Automated Disulfide Bond Detection
#
# Let's analyze all three-finger toxin structures automatically:

# %%
def find_disulfide_bonds(structure_path):
    """Find disulfide bonds in a protein structure."""
    print(f"🔍 Analyzing {Path(structure_path).name}...")

    # Load structure
    structure = gemmi.read_structure(str(structure_path))

    # Find all cysteine sulfur atoms
    cysteines = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == "CYS":
                    for atom in residue:
                        if atom.name == "SG":  # Sulfur atom
                            res_num = int(residue.seqid.num)
                            chain_id = str(chain.name)
                            position = np.array(atom.pos.tolist())
                            cysteines.append((res_num, chain_id, position))
                            break

    # Find close pairs (disulfide bonds)
    bonds = []
    for i in range(len(cysteines)):
        for j in range(i + 1, len(cysteines)):
            res1, chain1, pos1 = cysteines[i]
            res2, chain2, pos2 = cysteines[j]

            distance = np.linalg.norm(pos1 - pos2)
            if distance <= 2.3:  # Typical S-S bond distance
                bonds.append((res1, chain1, res2, chain2))
                print(f"  💚 Found bond: {chain1}:{res1} ↔ {chain2}:{res2} ({distance:.1f}Å)")

    return bonds

# Analyze all toxin structures
!mkdir -p "{tmp_path}disulfide"

structures = ["cobra.cif", "krait.cif", "snake.cif"]
all_bonds = {}

for structure in structures:
    structure_path = f"{data_path}3Ftx/{structure}"
    bonds = find_disulfide_bonds(structure_path)
    all_bonds[structure] = bonds
    print(f"🎯 {structure}: Found {len(bonds)} disulfide bonds!")
    print()

print(f"✅ Analyzed {len(structures)} structures!")

# %% [markdown]
# ## 📝 Generate Annotation Files
#
# Automatically create TOML annotation files for all structures:

# %%
def write_disulfide_annotations(structure_name, bonds, output_file):
    """Write disulfide bond annotations to a TOML file."""
    with open(output_file, 'w') as f:
        f.write(f"# Disulfide bond annotations for {structure_name}\n")
        f.write("# Generated automatically by FlatProt\n\n")

        for i, (res1, chain1, res2, chain2) in enumerate(bonds):
            f.write("[[annotations]]\n")
            f.write("type = \"line\"\n")
            f.write(f"indices = [\"{chain1}:{res1}\", \"{chain2}:{res2}\"]\n")
            f.write("[annotations.style]\n")
            f.write("line_color = \"#32CD32\"     # Lime green\\n")
            f.write("stroke_width = 2.0\n")
            f.write("line_style = [4, 2]        # Dashed line\n")
            f.write("connector_radius = 0.5\n\n")

    print(f"📝 Created {output_file} with {len(bonds)} disulfide bonds")

# Create annotation files for all structures
for structure in structures:
    if all_bonds[structure]:  # Only if bonds were found
        base_name = structure.replace(".cif", "")
        annotation_file = f"{tmp_path}disulfide/{base_name}_disulfide.toml"
        write_disulfide_annotations(structure, all_bonds[structure], annotation_file)

# %% [markdown]
# ## 🎯 Align Structures for Consistent Orientation
#
# First align all structures to ensure comparable visualizations:
# *Note: The first time you run this, it will take a while to download the database.*
# %%
# Align all structures to reference database
print("🔄 Aligning structures for consistent orientation...")
for structure in structures:
    if all_bonds[structure]:  # Only if bonds were found
        base_name = structure.replace(".cif", "")
        input_file = f"{data_path}3Ftx/{structure}"
        matrix_file = f"{tmp_path}disulfide/{base_name}_matrix.npy"
        info_file = f"{tmp_path}disulfide/{base_name}_info.json"

        !uv run flatprot align "{input_file}" "{matrix_file}" "{info_file}" --min-probability 0.5 --quiet
        print(f"✅ Aligned {base_name}")

print("\n🎯 All structures aligned!")

# %% [markdown]
# ## 🎨 Create Aligned Projections with Disulfide Bonds
#
# Generate visualizations using alignment matrices for consistent orientation:

# %%
# Create aligned projections for all structures with disulfide bonds
for structure in structures:
    if all_bonds[structure]:  # Only if bonds were found
        base_name = structure.replace(".cif", "")
        input_file = f"{data_path}3Ftx/{structure}"
        output_file = f"{tmp_path}disulfide/{base_name}_disulfide.svg"
        annotation_file = f"{tmp_path}disulfide/{base_name}_disulfide.toml"
        matrix_file = f"{tmp_path}disulfide/{base_name}_matrix.npy"

        !uv run flatprot project "{input_file}" -o "{output_file}" --annotations "{annotation_file}" --matrix "{matrix_file}" --canvas-width 600 --canvas-height 500 --show-positions major

        print(f"🎨 Created {output_file}")

print("\\n✅ All aligned projections with disulfide bonds created!")

# %% [markdown]
# ## 🎉 View the Results

# %%
from IPython.display import HTML, display

# Display all results
print("🧬 Three-Finger Toxin Family with Disulfide Bonds:")
print()

for structure in structures:
    if all_bonds[structure]:
        base_name = structure.replace(".cif", "")
        svg_file = f"{tmp_path}disulfide/{base_name}_disulfide.svg"

        if Path(svg_file).exists():
            print(f"🐍 {base_name.title()} Toxin:")
            with open(svg_file) as f:
                svg = f.read()
            svg = svg.replace('<svg ', '<svg style="max-width: 100%; height: auto;" ', 1)
            display(HTML(f'<div style="text-align: center; margin: 20px; border: 1px solid #ddd; padding: 10px;">{svg}</div>'))
            print(f"💚 {len(all_bonds[structure])} disulfide bonds highlighted")
            print()
        else:
            print(f"❌ Could not create {base_name} projection")

print("🔗 Green dashed lines show stabilizing disulfide bonds")
print("🎯 Notice how all three toxins have similar disulfide patterns!")
print("🔄 All structures are consistently aligned for easy comparison")

# %% [markdown]
# ## 🧬 **About three-finger toxin disulfide bonds:**
# - **Conserved pattern** across cobra, krait, and snake toxins
# - **Three disulfide bonds** create the rigid three-finger structure
# - **Cysteine pairing** follows the characteristic pattern
# - **Structural stability** allows toxins to function in harsh environments
# - **Evolutionary conservation** shows functional importance

# %% [markdown]
# ## 💡 **The Power of Automation**
#
# This workflow demonstrates:
# - **Batch processing** of multiple protein structures
# - **Automated feature detection** using structural criteria
# - **Programmatic visualization** generation
# - **Reproducible analysis** for any protein family
#
# **Perfect for comparative structural biology!** 🎯✨
