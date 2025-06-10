#!/usr/bin/env -S uv --quiet run --script


import pymol
from pymol import cmd
import numpy as np


def launch_and_interact(structure_path):
    # Launch PyMOL
    pymol.finish_launching(["pymol", structure_path])

    print("\nAdjust the structure manually in PyMOL.")
    input(
        "\nPress Enter here once you finish adjusting in PyMOL and are ready to save the matrix..."
    )

    # Get the rotation matrix from PyMOL
    matrix = cmd.get_object_matrix("all")

    # Save matrix to file
    save_matrix(matrix, "rotation_matrix.npy")

    # Close PyMOL
    cmd.quit()


def save_matrix(matrix, filename):
    np_matrix = np.array(matrix).reshape(4, 4)
    np.save(filename, np_matrix)
    print(f"Rotation matrix saved as '{filename}'")


if __name__ == "__main__":
    structure_path = "data/3Ftx/cobra.cif"  # Replace with your structure file path
    launch_and_interact(structure_path)
