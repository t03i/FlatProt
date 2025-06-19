# %% [markdown]
# # FlatProt: UniProt to AlphaFold Visualization
#
# **From UniProt ID to 2D protein visualization in minutes!**
#
# This example shows how to automatically download AlphaFold structures,
# align them to protein families, and create publication-ready visualizations
# with functional annotations. AlphaFold mmCIF files include secondary structure
# information, so no separate DSSP file is needed!

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
# ## 🎯 Configure Target Protein
#
# Set the UniProt ID for the protein you want to analyze.
# We'll automatically download the AlphaFold structure and create annotations.

# %%
# =============================================================================
# 🎯 TARGET PROTEIN - CHANGE THIS TO ANY UNIPROT ID YOU WANT TO ANALYZE
# =============================================================================
UNIPROT_ID = "P69905"  # Human Hemoglobin subunit alpha
# =============================================================================

# %%
# Create output directory
!mkdir -p "{tmp_path}uniprot/{UNIPROT_ID}"

print("="*70)
print(f"🎯 TARGET PROTEIN: {UNIPROT_ID}")
print("="*70)
print(f"📁 Working directory: {tmp_path}uniprot/{UNIPROT_ID}/")

# Common examples for reference:
print(f"\n💡 Example proteins to try:")
print(f"   P69905 - Human Hemoglobin alpha (oxygen transport)")
print(f"   P02144 - Human Myoglobin (oxygen storage)")
print(f"   P04637 - Human p53 (tumor suppressor)")
print(f"   P01308 - Human Insulin (hormone)")
print("="*70)

# %% [markdown]
# ## 📥 Download AlphaFold Structure
#
# Automatically fetch the predicted structure from the AlphaFold database.

# %%
import requests
import json

# Download AlphaFold structure
af_url = f"https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v4.cif"
structure_file = f"{tmp_path}uniprot/{UNIPROT_ID}/AF-{UNIPROT_ID}-F1-model_v4.cif"

print(f"📥 Downloading AlphaFold structure...")
print(f"   URL: {af_url}")

!wget -q "{af_url}" -O "{structure_file}" || echo "❌ Download failed - check UniProt ID"

# Verify download
import os
if os.path.exists(structure_file) and os.path.getsize(structure_file) > 1000:
    print(f"✅ Structure downloaded: {os.path.getsize(structure_file):,} bytes")
else:
    print(f"❌ Download failed or file too small")
    print(f"   Check if UniProt ID '{UNIPROT_ID}' exists in AlphaFold database")

# %% [markdown]
# ## ✅ Secondary Structure Information
#
# AlphaFold mmCIF files already contain secondary structure information!

# %%
print(f"✅ Secondary structure: Included in AlphaFold mmCIF file")
print(f"   FlatProt will automatically use the embedded secondary structure data")
print(f"   No separate DSSP file needed!")

# %% [markdown]
# ## 🌐 Fetch UniProt Annotations
#
# Download functional annotations from UniProt to highlight important sites.

# %%
def fetch_uniprot_annotations(uniprot_id):
    """Fetch functional annotations from UniProt API."""
    print(f"🌐 Fetching annotations from UniProt...")

    # UniProt REST API URL
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        annotations = []

        # Extract features (binding sites, active sites, etc.)
        features = data.get('features', [])
        print(f"   Found {len(features)} total features")

        for feature in features:
            feature_type = feature.get('type', '')
            description = feature.get('description', '')
            location = feature.get('location', {})

            # Extract ligand information if available
            ligand_info = feature.get('ligand', {})
            ligand_name = ligand_info.get('name', '')

            # Create better labels
            if ligand_name:
                label = f"{ligand_name} binding"
                description = f"Binds {ligand_name}"
            elif description:
                label = description[:20] + "..." if len(description) > 20 else description
            else:
                label = feature_type.replace(' ', '_').title()

            # Handle location format (always has start/end with value)
            start_pos = None
            end_pos = None

            if 'start' in location and 'end' in location:
                start_data = location['start']
                end_data = location['end']

                if 'value' in start_data and 'value' in end_data:
                    start_pos = start_data['value']
                    end_pos = end_data['value']

            # Only process features with valid positions
            if start_pos is not None and end_pos is not None:
                # Map feature types to annotation types and colors
                if feature_type in ['Binding site', 'Active site', 'Site', 'Metal binding']:
                    if start_pos == end_pos:  # Single residue
                        color = '#E74C3C' if feature_type == 'Active site' else '#9B59B6'
                        annotations.append({
                            'type': 'point',
                            'label': label,
                            'index': f"A:{start_pos}",
                            'color': color,
                            'description': description
                        })
                        print(f"   + Point: {feature_type} at A:{start_pos} - {label}")
                    else:  # Range - must be at least 3 residues for areas
                        area_size = end_pos - start_pos + 1
                        if area_size >= 3:
                            annotations.append({
                                'type': 'area',
                                'label': label,
                                'range': f"A:{start_pos}-{end_pos}",
                                'color': '#F39C12',
                                'description': description
                            })
                            print(f"   + Area: {feature_type} at A:{start_pos}-{end_pos} - {label} ({area_size} residues)")
                        else:
                            print(f"   - Skipped: {feature_type} at A:{start_pos}-{end_pos} - too small ({area_size} residues)")

                # Skip domains entirely - they are usually too large and not specific functional sites

        print(f"✅ Found {len(annotations)} functional annotations")
        return annotations

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error fetching UniProt data: {e}")
        return []
    except KeyError as e:
        print(f"⚠️ Unexpected UniProt data format: missing {e}")
        return []
    except Exception as e:
        print(f"⚠️ Could not fetch UniProt annotations: {e}")
        return []

# Fetch annotations
annotations = fetch_uniprot_annotations(UNIPROT_ID)

# %% [markdown]
# ## 📝 Create Annotation File
#
# Convert UniProt annotations to FlatProt format.

# %%
def create_annotation_file(annotations, output_file):
    """Create TOML annotation file from UniProt data."""

    if not annotations:
        print("ℹ️ No annotations to create")
        return False

    toml_content = "# Functional annotations from UniProt\n\n"

    for i, ann in enumerate(annotations):
        toml_content += f"[[annotations]]\n"
        toml_content += f"type = \"{ann['type']}\"\n"
        toml_content += f"label = \"{ann['label']}\"\n"

        if ann['type'] == 'point':
            toml_content += f"index = \"{ann['index']}\"\n"
        elif ann['type'] == 'area':
            toml_content += f"range = \"{ann['range']}\"\n"

        toml_content += f"[annotations.style]\n"
        toml_content += f"color = \"{ann['color']}\"\n"

        if ann['type'] == 'point':
            toml_content += f"marker_radius = 3\n"
            toml_content += f"label_font_size = 11\n"
        elif ann['type'] == 'area':
            toml_content += f"fill_color = \"{ann['color']}\"\n"
            toml_content += f"opacity = 0.3\n"
            toml_content += f"stroke_width = 1.5\n"

        toml_content += f"\n"

    # Write annotation file
    with open(output_file, 'w') as f:
        f.write(toml_content)

    print(f"✅ Created annotation file: {output_file}")
    return True

# Create annotations if we found any
annotations_file = f"{tmp_path}uniprot/{UNIPROT_ID}/annotations.toml"
has_annotations = create_annotation_file(annotations, annotations_file)

# %% [markdown]
# ## 🎨 Create Custom Style
#
# Define a modern color scheme for the visualization.

# %%
# Create custom style file
style_content = """
# Modern protein visualization style
[helix]
color = "#3498DB"        # Professional blue
stroke_color = "#2980B9"
stroke_width = 1.5
opacity = 0.8

[sheet]
color = "#E74C3C"        # Strong red
stroke_color = "#C0392B"
stroke_width = 1.5
opacity = 0.8

[coil]
stroke_color = "#7F8C8D" # Modern gray
stroke_width = 1.0
opacity = 0.9

[position_annotation]
show_residue_numbers = true
font_size = 10
color = "#2C3E50"
"""

style_file = f"{tmp_path}uniprot/{UNIPROT_ID}/style.toml"
with open(style_file, 'w') as f:
    f.write(style_content)

print(f"✅ Created style file: {style_file}")

# %% [markdown]
# ## 🔄 Align to Protein Family
#
# Use FlatProt's alignment feature to find the optimal rotation matrix.
# *Note: The first time you run this, it will take a while to download the database.*

# %%
# Generate alignment matrix for consistent orientation
matrix_file = f"{tmp_path}uniprot/{UNIPROT_ID}/{UNIPROT_ID}_matrix.npy"
info_file = f"{tmp_path}uniprot/{UNIPROT_ID}/{UNIPROT_ID}_alignment.json"

print(f"🔄 Aligning protein to family database...")

# Build alignment command (generates matrix, doesn't create new structure)
align_cmd = f'uv run flatprot align "{structure_file}" "{matrix_file}" "{info_file}"'
align_cmd += ' --min-probability 0.5 --quiet'

print(f"   Running: flatprot align...")

# Run alignment with proper error handling
alignment_success = True
try:
    result = !{align_cmd}
    # Check if files were actually created
    if not os.path.exists(matrix_file) or not os.path.exists(info_file):
        alignment_success = False
except:
    alignment_success = False

# Check alignment result and provide informative feedback
if alignment_success and os.path.exists(matrix_file):
    print(f"✅ Alignment completed: {matrix_file}")
    # Try to read alignment info for additional details
    try:
        import json
        with open(info_file, 'r') as f:
            align_info = json.load(f)
            if 'probability' in align_info:
                prob = align_info['probability']
                print(f"   📊 Alignment probability: {prob:.3f}")
            if 'scop_id' in align_info:
                scop_id = align_info['scop_id']
                print(f"   🏷️  Matched family: {scop_id}")
    except:
        pass  # Continue without detailed info if parsing fails
    use_matrix = True
else:
    print(f"⚠️ Alignment failed - this protein may not match any family in the database")
    print(f"   📝 Note: Using default inertia-based orientation instead")
    print(f"   💡 The visualization will still work, just without family-specific alignment")
    use_matrix = False

# %% [markdown]
# ## 🎨 Generate Visualization
#
# Create the final 2D projection with all enhancements.

# %%
# Generate the final visualization
output_svg = f"{tmp_path}uniprot/{UNIPROT_ID}/{UNIPROT_ID}_visualization.svg"

print(f"🎨 Creating 2D visualization...")

# Build project command
project_cmd = f'uv run flatprot project "{structure_file}" "{output_svg}"'
project_cmd += f' --style "{style_file}"'

if has_annotations:
    project_cmd += f' --annotations "{annotations_file}"'

if use_matrix:
    project_cmd += f' --matrix "{matrix_file}"'

project_cmd += ' --show-positions major'
project_cmd += ' --canvas-width 800 --canvas-height 600'
project_cmd += ' --quiet'

print(f"   Running: flatprot project...")
!{project_cmd}

if os.path.exists(output_svg):
    print(f"✅ Visualization created: {output_svg}")
else:
    print(f"❌ Visualization failed")

# %% [markdown]
# ## 🖼️ Display Results

# %%
# Display the final visualization
try:
    from IPython.display import SVG, display
    if os.path.exists(output_svg):
        print(f"🎉 Displaying visualization for {UNIPROT_ID}:")
        display(SVG(output_svg))

        # Show summary
        print(f"\n📊 Summary:")
        print(f"   Protein: {UNIPROT_ID}")
        print(f"   Structure: AlphaFold prediction")
        print(f"   Secondary structure: ✅ Embedded in mmCIF file")
        if use_matrix:
            print(f"   Family alignment: ✅ Applied - protein matched database family")
        else:
            print(f"   Family alignment: ⚠️ Not applied - using default orientation")
        print(f"   Functional annotations: {'✅ ' + str(len(annotations)) + ' features' if annotations else '❌ None found'}")

    else:
        print(f"❌ No visualization file found")

except ImportError:
    print(f"📝 Visualization created successfully!")
    print(f"💡 In a Jupyter environment, you would see the SVG here.")
    print(f"📁 File saved to: {output_svg}")

# %% [markdown]
# ## 🔬 Create Additional Visualizations
#
# Generate different annotation levels for comparison - all using the same orientation.

# %%
# Create a version without annotations for comparison
basic_svg = f"{tmp_path}uniprot/{UNIPROT_ID}/{UNIPROT_ID}_basic.svg"

print(f"🔬 Creating comparison visualization (same alignment, no annotations)...")

basic_cmd = f'uv run flatprot project "{structure_file}" "{basic_svg}"'
basic_cmd += f' --style "{style_file}"'
if use_matrix:
    basic_cmd += f' --matrix "{matrix_file}"'
basic_cmd += ' --show-positions minimal --canvas-width 800 --canvas-height 600 --quiet'

!{basic_cmd}

# Create a minimal version (no annotations)
minimal_svg = f"{tmp_path}uniprot/{UNIPROT_ID}/{UNIPROT_ID}_minimal.svg"

print(f"🎯 Creating minimal visualization...")

minimal_cmd = f'uv run flatprot project "{structure_file}" "{minimal_svg}"'
minimal_cmd += f' --style "{style_file}"'
if use_matrix:
    minimal_cmd += f' --matrix "{matrix_file}"'
minimal_cmd += ' --show-positions none --canvas-width 800 --canvas-height 600 --quiet'

!{minimal_cmd}

print(f"✅ Created comparison visualizations")

# %% [markdown]
# ## 🖼️ Annotation Level Comparison
#
# Display different annotation levels using the same protein orientation.

# %%
# Display comparison of different visualization approaches
try:
    from IPython.display import HTML, display

    # Read SVG files - adjust descriptions based on whether alignment worked
    alignment_desc = "family-aligned" if use_matrix else "default orientation"

    visualizations = [
        ("🎨 Full Featured", output_svg, f"Complete view: {alignment_desc} + UniProt annotations + position labels"),
        ("🔄 Structure Only", basic_svg, f"Structure view: {alignment_desc} + minimal position labels (no annotations)"),
        ("⚪ Clean Minimal", minimal_svg, f"Minimal view: {alignment_desc} only (no annotations or position labels)")
    ]

    existing_viz = [(name, path, desc) for name, path, desc in visualizations if os.path.exists(path)]

    if existing_viz:
        print(f"🖼️ Displaying {len(existing_viz)} visualization variants:")

        # Create side-by-side display
        html_content = """
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin: 20px 0;">
        """

        for name, svg_path, description in existing_viz:
            # Read and resize SVG
            with open(svg_path, 'r') as f:
                svg_content = f.read()

            # Make SVGs smaller for comparison (all are now 800x600)
            svg_content = svg_content.replace('width="800"', 'width="300"')
            svg_content = svg_content.replace('height="600"', 'height="240"')

            html_content += f"""
            <div style="text-align: center; margin: 10px; max-width: 320px;">
                <h4 style="margin-bottom: 10px; color: #777;">{name}</h4>
                <div style="border: 2px solid #ECF0F1; border-radius: 8px; padding: 15px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {svg_content}
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 8px; line-height: 1.3;">{description}</p>
            </div>
            """

        html_content += "</div>"
        display(HTML(html_content))

    else:
        print(f"❌ No visualization files found for comparison")

except ImportError:
    print(f"📝 All visualizations created successfully!")
    print(f"\n📁 Generated files:")
    for name, path, desc in visualizations:
        if os.path.exists(path):
            print(f"   {name}: {path}")

# %% [markdown]
# ## 📊 Summary and Next Steps
#
# **What we accomplished:**
#
# ✅ **Automated workflow** - From UniProt ID to visualization in one notebook
# ✅ **AlphaFold integration** - Automatic structure download with built-in secondary structure
# ✅ **Family alignment** - Optimal orientation using FlatProt's database
# ✅ **Functional annotations** - UniProt-derived binding sites and domains
# ✅ **Multiple views** - Comparison of different visualization approaches
# ✅ **Publication ready** - High-quality SVG output with custom styling
# ✅ **No separate DSSP file** - Secondary structure embedded in AlphaFold mmCIF
#
#
# **📚 Documentation:**
# - UniProt API: https://www.uniprot.org/help/api
# - AlphaFold database: https://alphafold.ebi.ac.uk/
# - FlatProt documentation: https://t03i.github.io/FlatProt/

# %% [markdown]
# ## 🎯 Try Different Proteins
# Simply change the `UNIPROT_ID` variable and re-run the notebook!
