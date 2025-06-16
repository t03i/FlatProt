# PyMOL Matrix Extraction

The `get_matrix.py` script allows you to interactively orient a protein structure in PyMOL and extract the transformation matrix for use with flatprot's `project` command.

## Usage

```bash
uv run scripts/get_matrix.py <structure_file>
```

### Example

```bash
uv run scripts/get_matrix.py data/3Ftx/cobra.cif
```

## How it Works

1. **Launches PyMOL**: Opens PyMOL with the specified structure file
2. **Interactive Orientation**: Allows you to manually rotate, translate, and zoom the structure to your desired view
3. **Matrix Extraction**: Captures PyMOL's current view matrix using `cmd.get_view()`
4. **Coordinate System Conversion**: Applies proper transformations for PyMOL ↔ flatprot compatibility
5. **File Output**: Saves the matrix as `rotation_matrix.npy`
6. **Auto-quit**: PyMOL automatically closes after matrix extraction

## Matrix Format

The script extracts PyMOL's view matrix and converts it to flatprot's format:

- **PyMOL format**: 18-element tuple (rotation[9], translation[3], origin[3], clipping[2], orthoscopic[1])
- **flatprot format**: 4x3 numpy array where:
  - First 3 rows: 3x3 rotation matrix
  - Last row: 3-element translation vector

## Coordinate System Transformation

The script automatically handles coordinate system differences between PyMOL and flatprot:

### The Problem
- **PyMOL**: Uses a graphics coordinate system (Y-up, right-handed)
- **flatprot**: Uses a mathematical coordinate system
- **Result**: Direct matrix transfer causes mirrored/flipped rotations

### Current Status
The script now applies a **complete coordinate system transformation**:
1. **Extracts PyMOL view matrix** (camera coordinates)
2. **Inverts the matrix** to convert from camera to object transformation
3. **Applies Y-axis flip** for final coordinate system alignment
4. **PyMOL automatically quits** after matrix extraction

### Complete Transformation Process

**Step 1: Camera vs Object Transformation**
- **PyMOL**: `cmd.get_view()` returns a **camera matrix** (camera rotates around stationary object)
- **flatprot**: Expects an **object transformation matrix** (object rotates while camera stays fixed)
- **Solution**: **Invert the camera matrix** → `object_matrix = inverse(camera_matrix)`

**Step 2: Y-Axis Flip**
- **Issue**: Coordinate system handedness differences
- **Solution**: Apply Y-axis flip → `[x, y, z] → [x, -y, z]`

**Final Result**: PyMOL orientation perfectly matches flatprot projection

## Using the Extracted Matrix

After running the script, you can use the extracted matrix with flatprot:

```bash
uv run flatprot project <structure_file> output.svg --matrix rotation_matrix.npy
```

## Tips for Best Results

1. **Orient Carefully**: Take time to get the exact orientation you want in PyMOL
2. **Use Cartoon Representation**: The script automatically sets cartoon representation for better visualization
3. **Test the Result**: Always test the extracted matrix with flatprot to ensure the orientation matches your expectations

## Dependencies

- PyMOL (for interactive structure manipulation)
- NumPy (for matrix operations)

## Installation

Install PyMOL using homebrew:

```bash
brew install pymol
```

The script will check for PyMOL availability and provide installation instructions if needed.
