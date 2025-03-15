# Matrix File Format

FlatProt uses transformation matrices to control how protein structures are oriented in the 2D visualization. This document explains the format of these matrix files.

## Overview

A transformation matrix in FlatProt consists of:

-   A 3×3 rotation matrix
-   A 3-element translation vector

Together, these define how the protein structure is positioned and oriented in 3D space before projection to 2D.

## File Format

Matrix files should be saved as NumPy `.npy` files containing a 2D array with one of the following shapes:

1. **4×3 matrix** (preferred format):

    - First 3 rows: 3×3 rotation matrix
    - Last row: 1×3 translation vector

2. **3×4 matrix** (will be automatically transposed):

    - First 3 columns: 3×3 rotation matrix
    - Last column: 3×1 translation vector

3. **3×3 matrix** (rotation only):
    - The entire matrix is treated as a rotation matrix
    - Translation is assumed to be zero

## Creating Matrix Files

You can create a matrix file using NumPy:

```python
import numpy as np

# Create a rotation matrix (identity rotation)
rotation = np.eye(3)

# Create a translation vector (move 5 units along x-axis)
translation = np.array([5.0, 0.0, 0.0])

# Combine into a 4×3 matrix
matrix = np.vstack([rotation, translation])

# Save to a file
np.save("my_transformation.npy", matrix)
```

## Validation

FlatProt validates matrix files to ensure they have the correct dimensions and format. If a matrix file is invalid, an error message will be displayed explaining the issue.
