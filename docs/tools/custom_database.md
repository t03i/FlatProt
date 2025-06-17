# Creating Custom Databases

FlatProt allows you to create custom alignment databases from your own collection of protein structures. This is useful when you want to work with specific protein families, custom datasets, or structures not included in the default database.

> **⚠️ Important Updates**
>
> The custom database script now creates a **complete database directory** containing:
> - HDF5 alignment database (`alignments.h5`)
> - Foldseek database (`foldseek/db*`) - **NEW!**
> - Database info file (`database_info.json`) - **NEW!**
>
> **Updated Usage:**
> - Output is now a **directory** (not a single `.h5` file)
> - Requires **Foldseek** to be installed
> - Use `--database ./my_database_dir` (not `./my_database.h5`)

## Overview

A custom database contains:

- **Transformation matrices**: Inertia-based transformations for each structure
- **Structure metadata**: Names, source files, and optional metadata
- **HDF5 alignment database**: Compatible with FlatProt's alignment system
- **Foldseek database**: For fast structural searching and alignment
- **Database validation**: Info file for FlatProt compatibility checks

## Data Format Requirements

### Input Structure Files

FlatProt supports the following structure file formats:

- **PDB format** (`.pdb`, `.ent`)
- **CIF format** (`.cif`) - **Recommended**
- **Compressed files** (`.pdb.gz`, `.cif.gz`)

### Directory Structure

Organize your structure files in a directory:

```
my_structures/
├── protein1.cif
├── protein2.cif
├── family_a/
│   ├── member1.cif
│   └── member2.cif
└── family_b/
    ├── member3.pdb
    └── member4.pdb.gz
```

### Structure Requirements

Each structure file must contain:

- **C-alpha atoms**: Required for transformation calculations
- **Standard residues**: Non-standard residues are handled gracefully
- **Single or multiple chains**: All chains are processed
- **Complete structures**: Missing C-alpha atoms are skipped with warnings

## Database Creation Script

Use the provided script to create custom databases:

```bash
python scripts/create_custom_database.py <input_folder> <output_database_dir> [options]
```

### Basic Usage

```bash
# Create basic database
python scripts/create_custom_database.py ./my_structures ./my_custom_db

# With verbose output
python scripts/create_custom_database.py ./my_structures ./my_custom_db --verbose

# Save creation info
python scripts/create_custom_database.py ./my_structures ./my_custom_db --info-file db_info.json

# Specify Foldseek executable if not in PATH
python scripts/create_custom_database.py ./my_structures ./my_custom_db --foldseek-executable /path/to/foldseek
```

### Recommended Workflow: Manual Structure Rotation

The preferred approach is to **manually rotate structures** before database creation:

### Step 1: Transform Structures in PyMOL/ChimeraX

**Important**: Use `transform_selection` or `transform_object` to actually modify coordinates, not just the view:

```python
# PyMOL example - Transform coordinates, not just view
load protein1.pdb
load protein2.pdb

# Align structures (this DOES change coordinates)
align protein1, protein2

# Transform coordinates (NOT just rotate view)
# Create transformation matrix for 45° rotation around X-axis
transform_selection protein1, [1,0,0,0, 0,0.707,-0.707,0, 0,0.707,0.707,0, 0,0,0,1]

# Or use transform_object for the entire object
transform_object protein1, [1,0,0,0, 0,0.707,-0.707,0, 0,0.707,0.707,0, 0,0,0,1]

# Save with modified coordinates
save protein1_rotated.pdb, protein1
save protein2_rotated.pdb, protein2
```

**Alternative: Use matrix_copy after visual alignment**:
```python
# Visual alignment approach
load protein1.pdb
load protein2.pdb

# Visually align/rotate protein1 as desired using mouse/rotate commands
rotate x, 45, protein1  # This only changes view
rotate y, -30, protein1 # This only changes view

# Apply the view transformation to coordinates
matrix_copy protein1, protein1  # Copies view matrix to object matrix
save protein1_rotated.pdb, protein1
```

### Step 2: Create Database (No Additional Rotation)

```bash
# Use structures as-is (recommended)
python scripts/create_custom_database.py ./rotated_structures ./my_database
```

This approach ensures:

- **Foldseek alignment accuracy**: Structures are oriented exactly as intended
- **Consistent alignment**: Manual control over structure orientation
- **Optimal visualization**: Structures positioned for best comparative viewing

## Alternative Rotation Methods

For special cases, programmatic rotation is available:

#### No Rotation (Default - Recommended)

```bash
python scripts/create_custom_database.py ./structures ./my_database --rotate-method none
```

Uses structures in their saved orientation. **Recommended for manually rotated structures**.

#### Random Rotation

```bash
python scripts/create_custom_database.py ./structures ./my_database --rotate-method random
```
Applies random rotations before transformation. Mainly useful for testing robustness.

### Naming Patterns

Control how structure names are generated:

#### Filename-based (Default)

```bash
python scripts/create_custom_database.py ./structures ./my_database --name-pattern filename
```

Uses the filename without extension as the structure name.

#### Parent Folder + Filename

```bash
python scripts/create_custom_database.py ./structures ./my_database --name-pattern parent_folder
```

Combines parent folder name with filename, useful for organizing families.

## Complete Example

### Manual Rotation Workflow

```bash
# Step 1: Manually rotate structures in PyMOL
pymol -c rotate_structures.py  # Your rotation script

# Step 2: Create database from pre-rotated structures
python scripts/create_custom_database.py \
    ./manually_rotated_structures \
    ./databases/my_custom_db \
    --name-pattern parent_folder \
    --info-file my_db_info.json \
    --log-file creation.log \
    --verbose
```

This workflow:

- Uses manually rotated structures (optimal for Foldseek alignment)
- Names entries using folder + filename pattern
- Saves detailed creation info and logs
- Provides verbose progress output
- Creates both HDF5 alignment database and Foldseek database
- Generates required database validation files

### PyMOL Batch Transformation Script

```python
# transform_structures.py - Example batch coordinate transformation script
import pymol
from pymol import cmd
import os
import math

def create_rotation_matrix(axis, angle_deg):
    """Create rotation matrix for given axis and angle."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    if axis == 'x':
        return [1,0,0,0, 0,cos_a,-sin_a,0, 0,sin_a,cos_a,0, 0,0,0,1]
    elif axis == 'y':
        return [cos_a,0,sin_a,0, 0,1,0,0, -sin_a,0,cos_a,0, 0,0,0,1]
    elif axis == 'z':
        return [cos_a,-sin_a,0,0, sin_a,cos_a,0,0, 0,0,1,0, 0,0,0,1]

# Load structures
structure_dir = "./original_structures"
output_dir = "./transformed_structures"

for filename in os.listdir(structure_dir):
    if filename.endswith(('.pdb', '.cif')):
        name = filename.split('.')[0]

        # Load structure
        cmd.load(f"{structure_dir}/{filename}", name)

        # Apply coordinate transformations (customize as needed)
        x_rotation_matrix = create_rotation_matrix('x', 45)
        y_rotation_matrix = create_rotation_matrix('y', -30)

        cmd.transform_object(name, x_rotation_matrix)
        cmd.transform_object(name, y_rotation_matrix)

        # Save transformed structure
        cmd.save(f"{output_dir}/{filename}", name)
        cmd.delete(name)

cmd.quit()
```

### Alternative: Interactive Transformation

```python
# interactive_transform.py - Transform after visual positioning
import pymol
from pymol import cmd

# Load structure
cmd.load("protein.pdb", "prot")

# Rotate visually to desired orientation using PyMOL GUI
# Then run this to apply the transformation to coordinates:
cmd.matrix_copy("prot", "prot")
cmd.save("protein_transformed.pdb", "prot")
```

## Using Custom Databases

Once created, use your custom database with FlatProt commands. The custom database creates a directory containing both the HDF5 alignment database and the Foldseek database.

### Database Structure

The script creates this directory structure:

```
my_custom_db/
├── alignments.h5          # HDF5 alignment database
├── database_info.json     # Database metadata and validation
└── foldseek/
    ├── db                 # Foldseek database files
    ├── db.index           # (created by Foldseek)
    ├── db.dbtype          # (created by Foldseek)
    └── ...                # (other Foldseek files)
```

### Using with FlatProt Commands

**Option 1: Command line argument (Recommended)**

```bash
# Specify the database directory path
flatprot align structure.cif --database ./databases/my_custom_db

# For other commands
flatprot project structure.cif output.svg --database ./databases/my_custom_db
flatprot overlay struct1.cif struct2.cif --database ./databases/my_custom_db
```

**Option 2: Environment variable**

```bash
# Set environment variable to database directory (not the .h5 file)
export FLATPROT_ALIGNMENT_DB_PATH="./databases/my_custom_db"
flatprot align structure.cif

# Or use it with other commands
flatprot project structure.cif output.svg
```

### Verify Database

Check your database contents:

```python
from flatprot.alignment.db import AlignmentDatabase

# Path to the HDF5 file within the database directory
with AlignmentDatabase("./databases/my_custom_db/alignments.h5") as db:
    entries = db.list_entries()
    print(f"Database contains {len(entries)} entries:")
    for entry_id in entries[:5]:  # Show first 5
        entry = db.get_entry(entry_id)
        print(f"  {entry_id}: {entry.structure_name}")
```

### Validate Database

Check if your database is valid and has all required files:

```python
from flatprot.utils.database import validate_database
from pathlib import Path

db_path = Path("./databases/my_custom_db")
if validate_database(db_path):
    print("✓ Database is valid and ready to use")
else:
    print("✗ Database validation failed - check required files")
```

## Database Format Details

### Complete Database Structure

The custom database consists of three components:

#### 1. HDF5 Alignment Database (`alignments.h5`)

```
alignments.h5
├── /entries/           # Group containing all entries
│   ├── entry_id_1/     # Group for each structure
│   │   ├── rotation_matrix    # 4x4 transformation matrix
│   │   └── metadata          # Optional metadata attributes
│   └── entry_id_2/
│       ├── rotation_matrix
│       └── metadata
└── /index/             # Group for fast lookups
    └── structure_names # Mapping structure names to entry IDs
```

#### 2. Foldseek Database (`foldseek/db*`)

Created by `foldseek createdb`, includes:
- `db` - Main database file
- `db.index` - Index for fast searching
- `db.dbtype` - Database type information
- Additional Foldseek-specific files

#### 3. Database Info (`database_info.json`)

Contains metadata and validation information:
```json
{
  "database_type": "custom_alignment",
  "version": "1.0.0",
  "creation_date": "2025-01-XX...",
  "description": "Custom FlatProt alignment database",
  "statistics": { ... },
  "files": {
    "alignment_database": "alignments.h5",
    "foldseek_database": "foldseek/db"
  }
}
```

### Transformation Matrices

Each entry contains a 4x4 transformation matrix that:

- Centers the structure at the origin
- Aligns it to principal inertia axes
- Can be applied to transform coordinates

### Metadata Storage

The script stores metadata for each entry:

- **source_file**: Original structure file path
- **rotation_method**: Rotation method used
- **file_size**: Original file size in bytes

## Advanced Configuration

### Manual Rotation Benefits

**Why manual rotation is preferred:**

1. **Foldseek Accuracy**: Structures are oriented exactly as intended for alignment
2. **Visual Control**: You can see exactly how structures will be positioned
3. **Domain-Specific**: Rotate to highlight specific structural features
4. **Consistent Orientation**: Ensure all family members have the same orientation
5. **Quality Control**: Inspect each structure during the rotation process

### PyMOL Transformation Examples

```python
# Example 1: Align all structures to a reference (coordinates changed)
cmd.load("reference.pdb", "ref")
for structure in ["protein1.pdb", "protein2.pdb"]:
    name = structure.split('.')[0]
    cmd.load(structure, name)
    cmd.align(name, "ref")  # This DOES change coordinates
    cmd.save(f"aligned_{structure}", name)

# Example 2: Transform to show active site face-on
cmd.load("enzyme.pdb", "enz")
# Create 90° Y-rotation matrix
y_90_matrix = [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,0,1]
cmd.transform_object("enz", y_90_matrix)
cmd.save("enzyme_transformed.pdb", "enz")

# Example 3: Interactive approach - visual then coordinate transformation
cmd.load("membrane_protein.pdb", "mem")
# Manually rotate in GUI to desired orientation, then:
cmd.matrix_copy("mem", "mem")  # Apply view to coordinates
cmd.save("membrane_protein_oriented.pdb", "mem")
```

### ChimeraX Alternative

ChimeraX also provides coordinate transformation commands:

```bash
# ChimeraX command line
open protein.pdb
turn x 45  # Rotate coordinates, not just view
save protein_rotated.pdb
```

### Batch Processing

For large datasets, consider:

- Processing in chunks if memory is limited
- Using the log file to monitor progress
- Running with `--verbose` to track issues

## Troubleshooting

### Common Issues

**No structures found**

- Check file extensions are supported
- Verify directory path is correct
- Files may be corrupted or empty

**Transformation calculation fails**

- Structure may lack C-alpha atoms
- File format may be invalid
- Try with `--verbose` for detailed error messages

**Memory issues**

- Process smaller batches
- Check available system memory
- Large structures require more memory

### Performance Tips

- **CIF format** is generally faster than PDB
- **Uncompressed files** process faster than compressed
- **SSD storage** improves I/O performance
- **Multiple small files** are faster than few large files

## Integration with FlatProt

Custom databases work seamlessly with all FlatProt commands:

```bash
# Project structures using custom database
flatprot project my_structure.cif output.svg --database ./my_database_dir

# Align structures
flatprot align structure.cif --database ./my_database_dir

# Create overlays
flatprot overlay structure1.cif structure2.cif --database ./my_database_dir --output overlay.svg

# Using environment variable
export FLATPROT_ALIGNMENT_DB_PATH="./my_database_dir"
flatprot project my_structure.cif output.svg
```

The custom database provides the same functionality as the default database, optimized for your specific protein collection.

## Requirements

Before creating custom databases, ensure you have:

- **Foldseek** installed and accessible (either in PATH or specify with `--foldseek-executable`)
- **Python dependencies**: `gemmi`, `numpy`, `flatprot`
- **Structure files** in supported formats (PDB, CIF)
- **Sufficient disk space** for both input structures and database files

## Troubleshooting

### Foldseek Not Found

```bash
# If Foldseek is not in PATH, specify the executable
python scripts/create_custom_database.py ./structures ./my_db \
    --foldseek-executable /path/to/foldseek
```

### Database Validation Failed

Use the validation function to check what's missing:

```python
from flatprot.utils.database import validate_database, REQUIRED_DB_FILES

db_path = Path("./my_database")
if not validate_database(db_path):
    print("Missing files:")
    for file in REQUIRED_DB_FILES:
        file_path = db_path / file
        if not file_path.exists():
            print(f"  - {file}")
```

### Memory Issues

For large collections:
- Process structures in smaller batches
- Use `--verbose` to monitor progress
- Ensure sufficient disk space for temporary files
