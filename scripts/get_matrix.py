#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "numpy"
# ]
# ///

import numpy as np
import sys
from pathlib import Path


def create_pymol_script_and_launch(structure_path):
    """Create a PyMOL script and launch PyMOL GUI for interactive use."""
    import tempfile
    import subprocess

    # Create output path for matrix
    output_matrix_path = Path.cwd() / "rotation_matrix.npy"

    # Create PyMOL script content
    script_content = f"""
# Load the structure
load {structure_path}, structure

# Show the structure in a nice view
show cartoon, structure
orient structure
zoom structure

# Set nice colors
color cyan, structure

# Print instructions to user
print "="*60
print "PyMOL Matrix Extraction - Interactive Mode"
print "="*60
print "The structure has been loaded and oriented."
print "Use mouse to rotate, translate, and zoom to get your desired view:"
print "  - Left mouse: rotate"
print "  - Middle mouse: zoom"
print "  - Right mouse: translate"
print ""
print "When you have the perfect orientation, type:"
print "  save_matrix"
print ""
print "To quit without saving:"
print "  quit"
print "="*60

# Define the save_matrix command
python
def save_matrix():
    import numpy as np
    from pymol import cmd

    try:
        # Get the current view matrix
        view_matrix = cmd.get_view()

        print(f"PyMOL view matrix (first 12 values): {{view_matrix[:12]}}")

        # Extract rotation matrix and translation from view matrix
        rotation_matrix = np.array(view_matrix[0:9]).reshape(3, 3)
        translation_vector = np.array(view_matrix[9:12])

        print(f"Extracted rotation matrix:\\n{{rotation_matrix}}")
        print(f"Extracted translation vector: {{translation_vector}}")

        # Convert to flatprot format (4x3 matrix)
        flatprot_matrix = np.vstack([rotation_matrix, translation_vector])

        # Apply camera-to-object transformation for PyMOL → flatprot compatibility
        # PyMOL get_view() returns camera matrix, but flatprot needs object transformation
        print("\\nApplying camera-to-object transformation...")
        print("PyMOL rotates camera around object, flatprot rotates object around camera.")

        # Extract rotation and translation from camera matrix
        camera_rotation = flatprot_matrix[:3, :]
        camera_translation = flatprot_matrix[3, :]

        print(f"Camera matrix (PyMOL view):\\n{{flatprot_matrix}}")

        # Convert to 4x4 homogeneous matrix for inversion
        camera_4x4 = np.eye(4)
        camera_4x4[:3, :3] = camera_rotation
        camera_4x4[:3, 3] = camera_translation

        try:
            # Invert camera matrix to get object transformation
            object_4x4 = np.linalg.inv(camera_4x4)

            # Convert back to flatprot's 4x3 format
            object_rotation = object_4x4[:3, :3]
            object_translation = object_4x4[:3, 3]
            object_matrix = np.vstack([object_rotation, object_translation])

            print(f"Object matrix (after inversion):\\n{{object_matrix}}")

            # Apply Y-axis flip for final coordinate system alignment
            print("\\nApplying Y-axis flip correction...")
            flip_y = np.diag([1, -1, 1])

            # Apply Y-flip to rotation part
            corrected_rotation = flip_y @ object_rotation
            corrected_matrix = np.vstack([corrected_rotation, object_translation])

            print(f"Final corrected matrix (with Y-flip):\\n{{corrected_matrix}}")

            # Use the fully corrected matrix
            flatprot_matrix = corrected_matrix
            print("✓ Camera-to-object transformation + Y-flip applied.")

        except np.linalg.LinAlgError as e:
            print(f"⚠️  Warning: Could not invert camera matrix: {{e}}")
            print("Using original matrix without inversion.")
            # Keep original matrix if inversion fails

        # Validate the matrix format
        try:
            # Try to validate using flatprot if available
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from flatprot.transformation.transformation_matrix import TransformationMatrix
            test_transform = TransformationMatrix.from_array(flatprot_matrix)
            print("✓ Matrix format validated successfully")
        except Exception as e:
            print(f"⚠️  Warning: Matrix format validation failed: {{e}}")
            print("The matrix will still be saved, but may need manual adjustment.")

        # Save matrix
        np.save("{output_matrix_path}", flatprot_matrix)

        print(f"\\n✓ Matrix saved to: {output_matrix_path}")
        print(f"Matrix shape: {{flatprot_matrix.shape}}")
        print("\\nYou can now use this matrix with:")
        print(f"  uv run flatprot project {structure_path} output.svg --matrix {output_matrix_path}")
        print("\\nMatrix extraction complete!")
        print("\\nClosing PyMOL...")
        cmd.quit()

        return flatprot_matrix

    except Exception as e:
        print(f"Error saving matrix: {{e}}")
        return None

# Register the command
cmd.extend("save_matrix", save_matrix)
python end

# Show help again
print ""
print "Ready! Adjust the view and type 'save_matrix' when done."
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        print(f"Loading structure: {structure_path}")
        print("Launching PyMOL GUI...")
        print("(This will open a new PyMOL window)")

        # Launch PyMOL with the script using subprocess to avoid threading issues
        cmd = ["pymol", script_path]
        subprocess.run(cmd, check=True)

        # Check if matrix was created
        if output_matrix_path.exists():
            print("\n✓ Matrix successfully saved!")

            # Load and display the matrix
            matrix = np.load(output_matrix_path)
            print(f"Final matrix shape: {matrix.shape}")
            print("Final matrix content:")
            print(matrix)

            return True
        else:
            print("\n⚠️  Matrix file was not created.")
            print("Make sure you typed 'save_matrix' in PyMOL before closing.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error running PyMOL: {e}")
        return False
    except FileNotFoundError:
        print("Error: PyMOL command not found.")
        print("Make sure PyMOL is installed and accessible from the command line.")
        print("On macOS with homebrew: brew install pymol")
        return False
    finally:
        # Clean up temporary script
        Path(script_path).unlink(missing_ok=True)


def launch_and_interact(structure_path):
    """Launch PyMOL GUI for interactive matrix extraction."""
    # Check if PyMOL is available (command line access)
    try:
        import subprocess

        subprocess.run(["pymol", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: PyMOL command line tool is not available.")
        print("Please install PyMOL: brew install pymol")
        return False

    # Use subprocess approach to avoid GUI threading issues
    return create_pymol_script_and_launch(structure_path)


def save_matrix(matrix, filename):
    """Save matrix in the format expected by flatprot."""
    np.save(filename, matrix)
    print(f"Transformation matrix saved as '{filename}'")
    print(f"Matrix shape: {matrix.shape}")
    print("Matrix content:")
    print(matrix)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python get_matrix.py <structure_file>")
        print("Example: python get_matrix.py data/3Ftx/cobra.cif")
        sys.exit(1)

    structure_path = sys.argv[1]

    # Validate structure file exists
    if not Path(structure_path).exists():
        print(f"Error: Structure file '{structure_path}' not found.")
        sys.exit(1)

    success = launch_and_interact(structure_path)

    if success:
        print("\n" + "=" * 50)
        print("SUCCESS: Matrix extraction completed!")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("Matrix extraction was not completed.")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
