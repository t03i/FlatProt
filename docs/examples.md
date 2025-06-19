# Examples

Interactive Jupyter notebooks demonstrating FlatProt's capabilities, from basic protein projections to advanced overlay visualizations.

## 🚀 Getting Started

### Quick Start

- **New to FlatProt?** Start with the Basic Projection example
- **Want to compare proteins?** Try the Protein Alignment example
- **Need publication graphics?** Use the Protein Overlay example

### Running Examples

- **Google Colab:** Click any badge below - setup is automatic!
- **Local Jupyter:** Clone the repo and run notebooks directly
- **Command Line:** Extract the core commands marked with 🎯

## 📖 Core Examples

### Basic Protein Projection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/project.ipynb)

**Perfect for beginners!** Ultra-clean example using native Jupyter shell commands (`!flatprot project`). Creates 2D visualizations with minimal complexity.

### Protein Family Overlay
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/overlay.ipynb)

**One command, multiple proteins!** Ultra-simplified demonstration using shell commands. Shows automatic clustering, alignment, and family visualization.

### Protein Alignment and Projection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/project_align.ipynb)

**Compare related protein structures with consistent alignment!** Shows how to align three similar toxins to a reference database and create side-by-side projections with consistent orientation.

### Protein Domain Splitting
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/split.ipynb)

**Extract and visualize protein domains separately!** Demonstrates how to use `flatprot split` to extract structural domains and create individual visualizations for comparative analysis.

## 🔬 Advanced Examples

### UniProt to AlphaFold Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/uniprot_alphafold.ipynb)

**From UniProt ID to visualization in minutes!** Automatically downloads AlphaFold structures, extracts functional annotations from UniProt, aligns to protein families, and creates publication-ready visualizations. Features automatic binding site detection and multiple visualization variants.

### Custom Styling and Annotations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/custom_styling.ipynb)

**Create protein visualizations with custom colors and annotations!** Demonstrates modern color schemes, point/line/area annotations, and style variations. Includes a side-by-side comparison gallery showing different aesthetic approaches.

### Disulfide Bond Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/disulfide.ipynb)

**Automated disulfide bond analysis!** Shows how to detect disulfide bonds, generate annotation files programmatically, and create annotated visualizations. Perfect example of structural bioinformatics automation.

## 💡 Learning Path

**🚀 Recommended (Clean & Simple):**
1. **Start:** [Basic Projection](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/project.ipynb) - Pure shell commands, minimal complexity
2. **Align:** [Protein Alignment and Projection](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/project_align.ipynb) - Structure comparison with consistent alignment
3. **Domains:** [Protein Domain Splitting](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/split.ipynb) - Extract and visualize domains separately
4. **Features:** [Disulfide Bond Detection](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/disulfide.ipynb) - Automated bond analysis + annotations
5. **Scale up:** [Protein Family Overlay](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/overlay.ipynb) - Multi-protein visualization

**🔬 Advanced (Full-Featured):**
6. **Customize:** [Custom Styling](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/custom_styling.ipynb) - Color schemes and annotations
7. **Research:** [UniProt to AlphaFold](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/uniprot_alphafold.ipynb) - Complete research workflow

## 🔧 Technical Notes

**About Notebook Complexity:** Much of the code in these notebooks handles automatic Google Colab setup (dependency installation, data download, environment configuration). The actual FlatProt usage is typically just 1-3 commands per example!

**Essential Commands (in Jupyter):**
```bash
# Basic projection (SVG output)
!flatprot project structure.cif -o output.svg

# Multi-protein overlay (PNG/PDF available)
!flatprot overlay "structures/*.cif" -o overlay.png --family 3000114 --clustering

# Structure alignment
!flatprot align structure.cif matrix.npy info.json
```

**Output Formats:**
- `flatprot project` → **SVG only** (vector graphics, perfect for notebooks)
- `flatprot overlay` → **SVG, PNG, PDF** (PNG great for Colab compatibility)
- `flatprot split` → **SVG only**

**Automation:** Notebooks are automatically generated from Python source files using `scripts/create-notebooks.sh` and jupytext.
