# %% [markdown]
# # FlatProt: Custom Styling and Annotations
#
# **Create protein visualizations with custom colors and annotations!**
#
# This example demonstrates how to apply custom styles and add various types of annotations
# (points, lines, areas) to highlight specific structural features.

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
# ## 🎨 Define Custom Styles
#
# Custom styles control the appearance of secondary structures (helices, sheets, coils).
# We'll create a vibrant color scheme inspired by modern design.
#
# 📖 **For complete style format reference, see:**
# https://t03i.github.io/FlatProt/file_formats/style/

# %%
# Create output directory
!mkdir -p "{tmp_path}custom_styling"

# Define modern custom styles
style_content = """
# Modern Protein Visualization Style
# Using vibrant, accessible colors

[helix]
color = "#FF6B6B"        # Coral red - warm and vibrant
stroke_color = "#E55555" # Darker coral for borders
stroke_width = 1.5
opacity = 0.85

[sheet]
color = "#4ECDC4"        # Teal - cool and calming
stroke_color = "#3BA39C" # Darker teal for borders
stroke_width = 1.5
opacity = 0.85

[coil]
stroke_color = "#95A5A6" # Modern gray
stroke_width = 1.0
opacity = 0.9

# Position annotation styling
[position_annotation]
show_residue_numbers = true
font_size = 10
font_family = "Arial, sans-serif"
color = "#2C3E50"        # Dark blue-gray
"""

# Write style file
style_file = f"{tmp_path}custom_styling/modern_style.toml"
with open(style_file, "w") as f:
    f.write(style_content)

print(f"✅ Created style file: {style_file}")

# %% [markdown]
# ## 🎯 Define Annotations
#
# Annotations highlight specific structural features:
# - **Point annotations**: Mark important residues (active sites, binding sites)
# - **Line annotations**: Show interactions or connections
# - **Area annotations**: Highlight structural regions or domains
#
# 📖 **For complete annotation format reference, see:**
# https://t03i.github.io/FlatProt/file_formats/annotations/

# %%
# Define comprehensive annotations for 1KT0
annotations_content = """
# Structural Annotations for 1KT0
# Highlighting key functional and structural features

# Active site residues - critical for enzyme function
[[annotations]]
type = "point"
label = "Active Site"
index = "A:67"
[annotations.style]
marker_radius = 3
color = "#E74C3C"        # Red for high importance
label_color = "#C0392B"
label_font_size = 12
label_offset = [8, -8]

# Binding pocket residue
[[annotations]]
type = "point"
label = "Binding Pocket"
index = "A:151"
[annotations.style]
marker_radius = 2.5
color = "#9B59B6"        # Purple for binding sites
label_color = "#8E44AD"
label_font_size = 11
label_offset = [6, 6]

# Catalytic triad connection
[[annotations]]
type = "line"
label = "Catalytic Network"
indices = ["A:67", "A:57"]
[annotations.style]
line_color = "#F39C12"   # Orange for interactions
stroke_width = 2.5
line_style = [6, 3]      # Dashed line
label_color = "#E67E22"
label_font_size = 10

# Structural loop of interest
[[annotations]]
type = "line"
label = "Flexible Loop"
indices = ["A:188", "A:195"]
[annotations.style]
line_color = "#3498DB"   # Blue for structural features
stroke_width = 2
label_color = "#2980B9"
label_font_size = 9

# Important structural region
[[annotations]]
type = "area"
label = "Catalytic Domain"
range = "A:189-195"
[annotations.style]
fill_color = "#2ECC71"   # Green for functional regions
opacity = 0.3
color = "#27AE60"        # Darker green border
stroke_width = 1.5
label_color = "#1E8449"
label_font_size = 11

# Secondary binding region
[[annotations]]
type = "area"
label = "Substrate Binding"
range = "A:140-160"
[annotations.style]
fill_color = "#F1C40F"   # Yellow for binding regions
opacity = 0.25
color = "#F39C12"        # Orange border
stroke_width = 1
label_color = "#D68910"
label_font_size = 10
"""

# Write annotations file
annotations_file = f"{tmp_path}custom_styling/protein_annotations.toml"
with open(annotations_file, "w") as f:
    f.write(annotations_content)

print(f"✅ Created annotations file: {annotations_file}")

# %% [markdown]
# ## 🚀 Generate Styled Visualization
#
# Now we'll use the `flatprot project` command with our custom style and annotation files.

# %%
# Input structure (1KT0 - a well-studied enzyme)
input_structure = f"{data_path}1KT0/1kt0.cif"
output_svg = f"{tmp_path}custom_styling/1kt0_custom_styled.svg"

# Generate the visualization
print("🎨 Creating custom styled protein visualization...")
!uv run flatprot project "{input_structure}" "{output_svg}" \
    --style "{style_file}" \
    --annotations "{annotations_file}" \
    --show-positions minimal \
    --canvas-width 1200 \
    --canvas-height 800 \
    --quiet

print(f"✅ Visualization saved to: {output_svg}")

# %% [markdown]
# ## 🖼️ Display Result

# %%
# Display the styled visualization
try:
    from IPython.display import SVG, display
    if Path(output_svg).exists():
        display(SVG(output_svg))
        print("🎉 Custom styled visualization displayed above!")
    else:
        print("❌ Output file not found")
except ImportError:
    print("📝 SVG created successfully! Open the file to view:")
    print(f"   {output_svg}")

# %% [markdown]
# ## 🎨 Create Alternative Color Schemes
#
# Let's create additional style variations to show different aesthetic approaches.

# %%
# Professional blue theme
professional_style = """
# Professional Blue Theme
[helix]
color = "#2E86AB"        # Professional blue
stroke_color = "#1F5F79"
stroke_width = 1.2
opacity = 0.8

[sheet]
color = "#A23B72"        # Deep magenta
stroke_color = "#7A2B56"
stroke_width = 1.2
opacity = 0.8

[coil]
stroke_color = "#677381"
stroke_width = 0.8
opacity = 0.9
"""

# Warm earth tones theme
earth_style = """
# Warm Earth Tones Theme
[helix]
color = "#D2691E"        # Chocolate/orange
stroke_color = "#A0522D"
stroke_width = 1.3
opacity = 0.8

[sheet]
color = "#8FBC8F"        # Dark sea green
stroke_color = "#6F8F6F"
stroke_width = 1.3
opacity = 0.8

[coil]
stroke_color = "#8B7355"  # Dark khaki
stroke_width = 0.9
opacity = 0.9
"""

# Save alternative styles
prof_style_file = f"{tmp_path}custom_styling/professional_style.toml"
earth_style_file = f"{tmp_path}custom_styling/earth_style.toml"

with open(prof_style_file, "w") as f:
    f.write(professional_style)

with open(earth_style_file, "w") as f:
    f.write(earth_style)

print("✅ Created alternative style files:")
print(f"   📘 Professional: {prof_style_file}")
print(f"   🌿 Earth tones: {earth_style_file}")

# %% [markdown]
# ## 🔄 Generate Style Variations

# %%
# Generate professional version
prof_output = f"{tmp_path}custom_styling/1kt0_professional.svg"
!uv run flatprot project "{input_structure}" "{prof_output}" \
    --style "{prof_style_file}" \
    --annotations "{annotations_file}" \
    --show-positions minimal \
    --quiet

# Generate earth tones version
earth_output = f"{tmp_path}custom_styling/1kt0_earth_tones.svg"
!uv run flatprot project "{input_structure}" "{earth_output}" \
    --style "{earth_style_file}" \
    --annotations "{annotations_file}" \
    --show-positions major \
    --quiet

print("✅ Generated style variations:")
print(f"   📘 Professional style: {prof_output}")
print(f"   🌿 Earth tones style: {earth_output}")

# %% [markdown]
# ## 🖼️ Style Comparison Gallery
#
# Let's display all three styles side by side to compare the visual differences.

# %%
# Display all three styles for comparison
try:
    from IPython.display import SVG, display, HTML
    import os

    # Check which files exist
    styles_to_display = [
        ("🎨 Modern Vibrant", output_svg),
        ("📘 Professional", prof_output),
        ("🌿 Earth Tones", earth_output)
    ]

    existing_files = [(name, path) for name, path in styles_to_display if os.path.exists(path)]

    if existing_files:
        print(f"📊 Displaying {len(existing_files)} style variations:")

        # Create HTML for side-by-side display
        html_content = """
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
        """

        for style_name, svg_path in existing_files:
            # Read SVG content
            with open(svg_path, 'r') as f:
                svg_content = f.read()

            # Modify SVG to have consistent size for comparison
            svg_content = svg_content.replace('width="1200"', 'width="350"')
            svg_content = svg_content.replace('height="800"', 'height="280"')

            html_content += f"""
            <div style="text-align: center; margin: 10px;">
                <h4 style="margin-bottom: 10px; color: #2C3E50;">{style_name}</h4>
                <div style="border: 2px solid #ECF0F1; border-radius: 8px; padding: 10px; background: white;">
                    {svg_content}
                </div>
            </div>
            """

        html_content += "</div>"

        # Display the comparison
        display(HTML(html_content))
        print("✨ Style comparison displayed above!")

    else:
        print("❌ No SVG files found for comparison")

except ImportError:
    print("📝 All SVG files created successfully!")
    print("💡 In a Jupyter environment, you would see a side-by-side comparison here.")
    print("\n📂 Generated files:")
    for name, path in styles_to_display:
        if os.path.exists(path):
            print(f"   {name}: {path}")

# %% [markdown]
# ## 🎯 Style Analysis
#
# **Comparing the three approaches:**
#
# - **🎨 Modern Vibrant**: Bold colors (coral/teal) for presentations and educational materials
# - **📘 Professional**: Subdued blues and magentas, perfect for publications and reports
# - **🌿 Earth Tones**: Warm, natural colors (orange/green) for a softer, organic feel
#
# Each style maintains the same functional annotations but creates a completely different visual impression!

# %% [markdown]
# ## 📊 Summary
#
# This example demonstrated:
#
# ✅ **Custom styling** - Modern color schemes for secondary structures
# ✅ **Point annotations** - Highlighting active sites and binding pockets
# ✅ **Line annotations** - Showing interactions and structural connections
# ✅ **Area annotations** - Marking functional domains and regions
# ✅ **Style variations** - Multiple aesthetic approaches
# ✅ **Position control** - Different levels of residue numbering
#
# **Next steps:**
# - Experiment with different color palettes
# - Add more specific annotations for your protein of interest
# - Combine with domain splitting for complex multi-domain proteins
# - Use in publications with the professional styling options
#
# **📚 Documentation:**
# - Style format reference: https://t03i.github.io/FlatProt/file_formats/style/
# - Annotation format reference: https://t03i.github.io/FlatProt/file_formats/annotations/
