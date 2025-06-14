#!/usr/bin/env -S uv --quiet run --script

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "matplotlib",
#   "biopython",
#   "numpy"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO
except ImportError as e:
    print(f"Error: Required package not available: {e}", file=sys.stderr)
    print("Install with: uv add biopython", file=sys.stderr)
    sys.exit(1)


def extract_sequence_from_structure(structure_file: Path, fasta_file: Path) -> None:
    """Extract protein sequence from structure file and save as FASTA."""
    file_extension = structure_file.suffix.lower()

    if file_extension in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    elif file_extension == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    structure = parser.get_structure("protein", str(structure_file))

    # Extract sequence from first chain
    sequences = []
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.get_id()[0] == " ":  # Standard amino acid
                    resname = residue.get_resname()
                    # Convert 3-letter to 1-letter amino acid code
                    aa_dict = {
                        "ALA": "A",
                        "CYS": "C",
                        "ASP": "D",
                        "GLU": "E",
                        "PHE": "F",
                        "GLY": "G",
                        "HIS": "H",
                        "ILE": "I",
                        "LYS": "K",
                        "LEU": "L",
                        "MET": "M",
                        "ASN": "N",
                        "PRO": "P",
                        "GLN": "Q",
                        "ARG": "R",
                        "SER": "S",
                        "THR": "T",
                        "VAL": "V",
                        "TRP": "W",
                        "TYR": "Y",
                    }
                    if resname in aa_dict:
                        sequence += aa_dict[resname]
                    else:
                        sequence += "X"  # Unknown residue

            if sequence:
                sequences.append((f"{chain.get_id()}", sequence))

    if not sequences:
        raise ValueError("No valid protein sequences found in structure")

    # Write FASTA file
    with open(fasta_file, "w") as f:
        for chain_id, seq in sequences:
            f.write(f">{structure_file.stem}_{chain_id}\n{seq}\n")


def convert_cif_to_pdb(cif_file: Path, pdb_file: Path) -> None:
    """Convert CIF file to PDB format using BioPython."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(cif_file))

    # Fix chain IDs that exceed PDB format limit (single character)
    chain_map = {}
    new_chain_id = ord("A")

    for model in structure:
        for chain in model:
            old_id = chain.get_id()
            if len(old_id) > 1 or old_id in chain_map.values():
                # Create a new single-character chain ID
                new_id = chr(new_chain_id)
                chain_map[old_id] = new_id
                chain.id = new_id
                new_chain_id += 1
                if new_chain_id > ord("Z"):
                    new_chain_id = ord("a")  # Use lowercase after Z

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))


def cleanup_ssdraw_cache():
    """Remove the SSDraw cache directory."""
    cache_dir = Path.home() / ".cache" / "ssdraw"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Removed SSDraw cache: {cache_dir}")


def main(
    structure_file: Path,
    output_file: Path,
    cleanup_cache: bool = False,
    ssdraw_path: Optional[Path] = None,
) -> None:
    """Load a structure and create a 2D diagram using SSDraw.

    This script uses SSDraw to create protein secondary structure diagrams.

    Parameters
    ----------
    structure_file
        Path to the input structure file (e.g., CIF, PDB). Must exist.
    output_file
        Path to save the output image (PNG format).
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract sequence from structure file
            fasta_file = temp_path / f"{structure_file.stem}.fasta"
            extract_sequence_from_structure(structure_file, fasta_file)

            # Check if SSDraw is available (try to import or run)
            try:
                # Use provided SSDraw path if available
                if ssdraw_path and ssdraw_path.exists():
                    ssdraw_script = ssdraw_path
                    print(f"Using provided SSDraw from: {ssdraw_script}")
                else:
                    # Fallback to cache-based approach
                    ssdraw_script = None

                    # 1. Check if SSDraw is in PATH or available as a module
                    result = subprocess.run(
                        ["python3", "-c", "import SSDraw"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        # SSDraw is available as a module, but we need the script
                        pass

                    # 2. Check for a persistent SSDraw installation in user cache
                    cache_dir = Path.home() / ".cache" / "ssdraw"
                    persistent_ssdraw = cache_dir / "SSDraw" / "SSDraw" / "SSDraw.py"

                    if persistent_ssdraw.exists():
                        ssdraw_script = persistent_ssdraw
                        print(f"Using cached SSDraw from: {ssdraw_script}")
                    else:
                        # 3. Clone to persistent cache location
                        print("SSDraw not found. Cloning to cache directory...")
                        cache_dir.mkdir(parents=True, exist_ok=True)

                        clone_cmd = [
                            "git",
                            "clone",
                            "https://github.com/ncbi/SSDraw.git",
                            str(cache_dir / "SSDraw"),
                        ]
                        clone_result = subprocess.run(
                            clone_cmd, capture_output=True, text=True
                        )
                        if clone_result.returncode != 0:
                            raise RuntimeError(
                                f"Failed to clone SSDraw: {clone_result.stderr}"
                            )

                        ssdraw_script = persistent_ssdraw
                        if not ssdraw_script.exists():
                            raise FileNotFoundError("SSDraw.py not found in repository")

                        print(f"Successfully cloned SSDraw to cache: {ssdraw_script}")

                    if not ssdraw_script:
                        raise FileNotFoundError("Could not locate SSDraw script")

                # Convert CIF to PDB if needed (SSDraw works better with PDB)
                if structure_file.suffix.lower() in [".cif", ".mmcif"]:
                    # Convert CIF to PDB format for better SSDraw compatibility
                    temp_pdb = temp_path / f"{structure_file.stem}.pdb"
                    convert_cif_to_pdb(structure_file, temp_pdb)
                    input_structure = temp_pdb
                else:
                    input_structure = structure_file

                # Run SSDraw
                ssdraw_cmd = [
                    "python3",
                    str(ssdraw_script),
                    "--fasta",
                    str(fasta_file),
                    "--name",
                    structure_file.stem,
                    "--pdb",
                    str(input_structure),
                    "--output",
                    str(output_file.with_suffix("")),  # SSDraw adds .png
                ]

                print(f"Running SSDraw command: {' '.join(ssdraw_cmd)}")
                ssdraw_result = subprocess.run(
                    ssdraw_cmd, capture_output=True, text=True
                )

                if ssdraw_result.returncode != 0:
                    print(f"SSDraw stdout: {ssdraw_result.stdout}")
                    print(f"SSDraw stderr: {ssdraw_result.stderr}")

                    # Check for common issues
                    stderr_text = ssdraw_result.stderr.lower()
                    if "mkdssp" in stderr_text or "dssp" in stderr_text:
                        raise RuntimeError(
                            "SSDraw requires DSSP (mkdssp) to be installed. Install with: brew install dssp (macOS) or apt-get install dssp (Ubuntu)"
                        )
                    elif "torch" in stderr_text:
                        raise RuntimeError(
                            "SSDraw requires PyTorch. Install with: pip install torch"
                        )
                    else:
                        raise RuntimeError(
                            f"SSDraw execution failed with code {ssdraw_result.returncode}"
                        )

                print("SSDraw execution completed successfully")

                # SSDraw typically adds .png extension, so move the file if needed
                generated_file = output_file.with_suffix("").with_suffix(".png")
                if generated_file.exists() and generated_file != output_file:
                    generated_file.rename(output_file)

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"SSDraw execution failed: {e}")
            except FileNotFoundError:
                raise RuntimeError(
                    "SSDraw not available. Please install from: https://github.com/ncbi/SSDraw"
                )

        print(f"Successfully created SSDraw diagram: {output_file}")

        # Clean up cache if requested
        if cleanup_cache:
            cleanup_ssdraw_cache()

    except Exception as e:
        print(f"Error processing structure {structure_file}: {e}", file=sys.stderr)
        # Clean up cache on error if requested
        if cleanup_cache:
            cleanup_ssdraw_cache()

        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a structure and create a 2D diagram using SSDraw."
    )
    parser.add_argument(
        "structure_file",
        type=Path,
        help="Path to the input structure file (e.g., CIF, PDB).",
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the output PNG image."
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Remove SSDraw cache after processing",
    )
    parser.add_argument(
        "--ssdraw-path",
        type=Path,
        help="Path to SSDraw.py script (if provided, skips auto-detection)",
    )

    try:
        args = parser.parse_args()
        if not args.structure_file.exists():
            print(
                f"Error: structure file not found: {args.structure_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        main(
            args.structure_file, args.output_file, args.cleanup_cache, args.ssdraw_path
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
