import numpy as np
from scipy.linalg import eigh
from Bio import PDB
from Bio.SeqUtils import molecular_weight
from Bio.PDB.Polypeptide import aa3

def get_residue_mass(residue_name):
    """
    Get the mass of an amino acid residue.
    
    :param residue_name: Three-letter code of the amino acid
    :return: Mass of the residue in kDa
    """
    
    # Calculate molecular weight
    mass = molecular_weight(residue_name, seq_type="protein")
    return mass / 1000  # Convert from Da to kDa

def read_pdb_for_inertia(pdb_file):
    """
    Read a PDB file and extract coordinates and masses for moment of inertia calculation.
    
    :param pdb_file: Path to the PDB file
    :return: coordinates (Nx3 array), masses (N-length array), structure object
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    coordinates = []
    masses = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    for atom in residue:
                        if atom.name == 'CA':  # We'll use only alpha carbons for simplicity
                            coordinates.append(atom.coord)
                            masses.append(get_residue_mass(residue.resname))
    
    return np.array(coordinates), np.array(masses), structure

def calculate_moment_of_inertia(coordinates, masses):
    """
    Calculate the moment of inertia tensor for a protein.
    
    :param coordinates: Nx3 array of atomic coordinates (x, y, z)
    :param masses: N-length array of atomic masses
    :return: 3x3 moment of inertia tensor, center of mass
    """
    # Calculate the center of mass
    total_mass = np.sum(masses)
    com = np.sum(coordinates * masses[:, np.newaxis], axis=0) / total_mass

    # Translate coordinates to center of mass
    coordinates_centered = coordinates - com

    # Calculate the moment of inertia tensor
    moment_of_inertia = np.zeros((3, 3))
    for coord, mass in zip(coordinates_centered, masses):
        moment_of_inertia[0, 0] += mass * (coord[1]**2 + coord[2]**2)
        moment_of_inertia[1, 1] += mass * (coord[0]**2 + coord[2]**2)
        moment_of_inertia[2, 2] += mass * (coord[0]**2 + coord[1]**2)
        moment_of_inertia[0, 1] -= mass * coord[0] * coord[1]
        moment_of_inertia[0, 2] -= mass * coord[0] * coord[2]
        moment_of_inertia[1, 2] -= mass * coord[1] * coord[2]

    # Fill in the symmetric elements
    moment_of_inertia[1, 0] = moment_of_inertia[0, 1]
    moment_of_inertia[2, 0] = moment_of_inertia[0, 2]
    moment_of_inertia[2, 1] = moment_of_inertia[1, 2]

    return moment_of_inertia, com

def orient_protein(moment_of_inertia):
    # Eigendecomposition of the moment of inertia tensor
    eigvals, eigvecs = eigh(moment_of_inertia)

    # Sort eigenvalues and eigenvectors
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Create rotation matrix
    rotation_matrix = eigvecs.T

    # Ensure right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1

    # Determine the orientation that has the least perturbation from identity matrix
    identity = np.eye(3)
    perturbations = [
        np.sum(identity * rotation_matrix),
        np.sum(identity * np.dot(rotation_matrix, np.diag([1, -1, -1]))),
        np.sum(identity * np.dot(rotation_matrix, np.diag([-1, 1, -1]))),
        np.sum(identity * np.dot(rotation_matrix, np.diag([-1, -1, 1])))
    ]

    best_orientation = np.argmax(perturbations)

    if best_orientation == 1:
        rotation_matrix = np.dot(rotation_matrix, np.diag([1, -1, -1]))
    elif best_orientation == 2:
        rotation_matrix = np.dot(rotation_matrix, np.diag([-1, 1, -1]))
    elif best_orientation == 3:
        rotation_matrix = np.dot(rotation_matrix, np.diag([-1, -1, 1]))

    return rotation_matrix

def apply_rotation_and_save(structure, rotation_matrix, com, output_pdb):
    """
    Apply rotation to the protein structure and save as a new PDB file.
    
    :param structure: Biopython Structure object
    :param rotation_matrix: 3x3 rotation matrix
    :param com: Center of mass
    :param output_pdb: Path to save the rotated structure
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Center the atom
                    centered_coord = atom.coord - com
                    # Apply rotation
                    rotated_coord = np.dot(rotation_matrix, centered_coord)
                    # Move back from center
                    atom.coord = rotated_coord + com

    # Save the rotated structure
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return output_pdb

def do_pymol_orient(pdb_file, output_pdb):
    coordinates, masses, structure = read_pdb_for_inertia(pdb_file)
    moment_of_inertia, com = calculate_moment_of_inertia(coordinates, masses)
    rotation_matrix = orient_protein(moment_of_inertia)
    apply_rotation_and_save(structure, rotation_matrix, com, output_pdb)
    return output_pdb