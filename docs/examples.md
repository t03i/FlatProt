# Examples

Interactive Jupyter notebooks demonstrating FlatProt's capabilities, from basic protein projections to advanced overlay visualizations.

**📝 Note:** The core examples use native Jupyter features (`!command`, `%%bash`) for maximum simplicity. In Google Colab, we use `%cd ..` after setup to align directory structures, eliminating path conditionals throughout the notebooks.

## 🚀 Getting Started

### Quick Start
- **New to FlatProt?** Start with the Simple Projection example
- **Want to compare proteins?** Try the 3FTX Alignment example
- **Need publication graphics?** Use the Protein Overlay example

### Running Examples
- **Google Colab:** Click any badge below - setup is automatic!
- **Local Jupyter:** Clone the repo and run notebooks directly
- **Command Line:** Extract the core commands marked with 🎯

## 📖 Core Examples

### Simple Projection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/simple_projection_v2.ipynb)

**Perfect for beginners!** Ultra-clean example using native Jupyter shell commands (`!flatprot project`). Creates beautiful 2D visualizations with minimal complexity.

### Protein Family Overlay
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/overlay_v2.ipynb)

**One command, multiple proteins!** Ultra-simplified demonstration using `%%bash` cells. Shows automatic clustering, alignment, and family visualization.

### Three-Finger Toxin Alignment
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/3ftx_simple.ipynb)

**Structure comparison made simple!** Clean shell command workflow for comparing three related toxin structures side-by-side.

## 🔬 Advanced Examples

### UniProt Integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/uniprot_projection.ipynb)

This notebook demonstrates how to fetch a protein structure from the AlphaFold Database (AFDB) using a UniProt ID, run DSSP to determine secondary structure, and then generate a 2D projection using FlatProt.

### Custom Annotations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/annotations.ipynb)

This notebook demonstrates how to apply custom styles and various annotations (point, line, area) to a FlatProt projection.


### Domain Analysis (Chainsaw)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/chainsaw.ipynb)

Generate and compare different 2D SVG visualizations of a protein structure based on its domains: Normal Projection, Domain-Aligned Projection, Domain-Separated Projection.

### Disulfide Bond Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/disulfide_auto.ipynb)

**Automated family analysis!** Batch processing of multiple structures to detect disulfide bonds, generate annotation files programmatically, and create comparative visualizations. Perfect example of structural bioinformatics automation.

### KLK Family Overlay (Advanced)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/klk_overlay.ipynb)

Advanced example showing complex overlay creation with automatic clustering, family alignment, and publication-quality visualizations of the Kallikrein protein family.

## 💡 Learning Path

**🚀 Recommended (Clean & Simple):**
1. **Start:** [Simple Projection](simple_projection_v2.ipynb) - Pure shell commands, minimal complexity
2. **Next:** [Three-Finger Toxin Alignment](3ftx_simple.ipynb) - Structure comparison made easy
3. **Features:** [Disulfide Bond Detection](disulfide_auto.ipynb) - Automated family analysis + annotations
4. **Scale up:** [Protein Family Overlay](overlay_v2.ipynb) - Multi-protein visualization

**🔬 Advanced (Full-Featured):**
5. **Customize:** [Custom Annotations](annotations.ipynb) & [Domain Analysis](chainsaw.ipynb)
6. **Research:** [UniProt Integration](uniprot_projection.ipynb) & [KLK Family Analysis](klk_overlay.ipynb)

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
