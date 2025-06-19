# FlatProt

<img width="910" alt="image" src=".github/images/family_fixed.png">

FlatProt is a tool for 2D protein visualization aimed at improving the comparability of protein structures through standardized 2D visualizations. FlatProt focuses on creating highly comparable representations for same-family proteins.

## 📚 Documentation

**[📖 Full Documentation](https://t03i.github.io/FlatProt/)** - Complete guide with examples, API reference, and tutorials

**[🔬 Interactive Examples](https://t03i.github.io/FlatProt/examples/)** - Jupyter notebooks with Google Colab integration

**[🧬 Try Now: UniProt to Visualization](https://colab.research.google.com/github/t03i/FlatProt/blob/main/examples/uniprot_alphafold.ipynb)** - From any UniProt ID to beautiful 2D visualization in minutes!

## 🚀 Quick Start

### Installation

FlatProt requires Python 3.11-3.13. Install using [uv](https://github.com/astral-sh/uv) (recommended):

**Option 1: Install and use persistently (recommended)**
```bash
# Install FlatProt (makes 'flatprot' command available)
uv tool add FlatProt

# Now you can use flatprot directly
flatprot --help
```

**Option 2: Run without installation**
```bash
# Run FlatProt without installing (use 'uvx flatprot' instead of 'flatprot')
uvx flatprot --help
```

### Dependencies

**Required:**
- [Foldseek](https://github.com/steineggerlab/foldseek) - for structure alignment
- [DSSP](https://pdb-redo.eu/dssp/download) - mkdssp version 4.4.0 (available via brew on macOS)

**Optional:**
- [Cairo](https://cairographics.org/) - for PNG/PDF output from overlay command

### Basic Usage

Generate a 2D protein visualization from a structure file:

```bash
# Option 1: AlphaFold structures (no DSSP needed - secondary structure included!)
flatprot project AF-P69905-F1-model_v4.cif --output protein_2d.svg

# Option 2: PDB/CIF files (add secondary structure first)
mkdssp your_protein.cif your_protein_with_dssp.cif
flatprot project your_protein_with_dssp.cif --output protein_2d.svg

# Note: If using without installation, replace 'flatprot' with 'uvx flatprot'
```

For detailed installation and usage instructions, see the [documentation](https://t03i.github.io/FlatProt/installation/).

## 📊 Example Visualizations

<img width="409" alt="Cobra protein visualization" src=".github/images/cobra.png">

<img width="409" alt="Protein overlay visualization" src=".github/images/overlay.png">

## Contributing

We welcome contributions to FlatProt! If you'd like to contribute, please follow these steps:

1.  **Fork the repository:** Create your own fork of the FlatProt repository on GitHub.
2.  **Create a branch:** Make your changes in a dedicated branch in your fork.
3.  **Submit a pull request:** Open a pull request from your branch to the `staging` branch of the t03i/FlatProt repository.

Please ensure your contributions adhere to the project's coding style and include tests where appropriate. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed guidelines.

## 📖 Citation & Data

If you use FlatProt in your research, please cite our preprint:

> **FlatProt: 2D visualization eases protein structure comparison**
> Tobias Olenyi, Constantin Carl, Tobias Senoner, Ivan Koludarov, Burkhard Rost
> *bioRxiv* 2025.04.22.650077; doi: [https://doi.org/10.1101/2025.04.22.650077](https://doi.org/10.1101/2025.04.22.650077)

**Datasets and supplementary data** are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15697296.svg)](https://doi.org/10.5281/zenodo.15697296)

## 🔧 CLI Commands

FlatProt provides four main commands:

- `flatprot project` - Create 2D SVG projections from protein structures
- `flatprot align` - Align protein structures using rotation
- `flatprot overlay` - Create overlay visualizations from multiple structures
- `flatprot split` - Extract and align structural regions for comparison

See the [CLI documentation](https://t03i.github.io/FlatProt/commands/project/) for detailed usage.

## 🔄 Common Workflows

**Note:** The following examples assume you've installed FlatProt with `uv tool add FlatProt`. If you're using the no-install option, replace `flatprot` with `uvx flatprot` in all commands.

### Single Structure Visualization
```bash
# AlphaFold structures (recommended - no preprocessing needed!)
flatprot project AF-P69905-F1-model_v4.cif --output protein_2d.svg

# Traditional PDB/CIF files (requires DSSP preprocessing)
mkdssp your_protein.cif your_protein_with_dssp.cif
flatprot project your_protein_with_dssp.cif --output protein_2d.svg
```

### Aligned Structure Visualization
```bash
# 1. Find optimal family alignment
flatprot align protein.cif -m alignment_matrix.npy

# 2. Create aligned 2D projection
flatprot project protein.cif --matrix alignment_matrix.npy -o aligned_protein.svg
```

### Family Comparison Workflow
```bash
# 1. Find optimal alignment for family
flatprot align reference_protein.cif -i family_info.json

# 2. Extract family ID from results
family_id=$(jq -r '.best_hit.target_id' family_info.json)

# 3. Create conserved overlay using fixed family ID
flatprot overlay "family_proteins/*.cif" --family "$family_id" -o family_overlay.png
```

### Domain Analysis Workflow
```bash
# 1. Create full structure view
flatprot project protein.cif -o full_structure.svg

# 2. Extract and align specific domains
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o domains.svg

# 3. Create domain overlay for comparison
flatprot overlay "domain_structures/*.cif" --family 3000114 -o domain_comparison.png
```

### Custom Alignment Workflow
```bash
# 1. Generate alignment matrix from first structure
flatprot align reference.cif -m alignment_matrix.npy

# 2. Apply same alignment to all structures
for file in *.cif; do
    flatprot project "$file" --matrix alignment_matrix.npy -o "aligned_${file%.cif}.svg"
done
```


## License

FlatProt is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
