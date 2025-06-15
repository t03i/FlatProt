#!/usr/bin/env python3
"""
FlatProt Runtime Performance Benchmark Script (Folder-based)

This script measures the runtime performance of FlatProt's project and align commands
on all protein structures in a given folder, tracking execution time and file sizes.
Based on the original notebook implementation from examples/runtime.

USAGE:
    python scripts/runtime_benchmark.py structures_folder [options]

QUICK START:
    # Basic benchmark (both project and align)
    python scripts/runtime_benchmark.py data/benchmark_structures

    # Project command only (faster)
    python scripts/runtime_benchmark.py data/benchmark_structures \\
        --skip-align

    # Limit to first 50 structures with custom results file
    python scripts/runtime_benchmark.py data/benchmark_structures \\
        --max-structures 50 \\
        --results-tsv small_benchmark_results.tsv

INPUT:
    Directory containing structure files (.cif, .pdb, .mmcif)
    The script will automatically find and process all structure files in the directory.

OPTIONS:
    --results-tsv PATH        TSV results file (default: runtime_results.tsv)
    --extensions EXT [EXT]    File extensions (default: .cif .pdb .mmcif)
    --max-structures INT      Max structures to process (default: all)
    --skip-project           Skip testing project command
    --skip-align             Skip testing align command
    --continue-on-error      Continue even if structures fail

OUTPUT FILES:
    - runtime_results.tsv: Benchmark results in tabular format only
    Note: SVG and matrix files are created temporarily for size measurement then deleted

RESULT COLUMNS:
    - protein_id: Extracted protein identifier
    - structure_file: Original structure filename
    - structure_file_path: Full path to structure file
    - protein_length: Number of residues
    - structure_file_size: Input file size in bytes
    - project_runtime: Project command (inertia-based) runtime in seconds
    - project_success: Whether project command succeeded
    - svg_file_size: Output SVG file size in bytes (from project)
    - project_error: Error message if project failed
    - align_runtime: Align + project with matrix combined runtime in seconds
    - align_success: Whether align+project commands succeeded
    - matrix_file_size: Output matrix file size in bytes
    - align_svg_file_size: Output SVG file size in bytes (from align+project)
    - align_error: Error message if align+project failed
    - timestamp: When the benchmark was run

EXAMPLE WORKFLOW:
    # 1. Setup structures using runtime_benchmark_setup.py
    python scripts/runtime_benchmark_setup.py human_proteome.tsv \\
        --output-dir data/human_proteome_200 \\
        --max-entries 200

    # 2. Run benchmark
    python scripts/runtime_benchmark.py data/human_proteome_200 \\
        --continue-on-error \\
        --results-tsv human_proteome_benchmark_results.tsv

    # 3. Analyze results
    head human_proteome_benchmark_results.tsv

PERFORMANCE NOTES:
    - Each structure takes ~1-10 seconds to process depending on size
    - Memory usage scales with protein size, typically <1GB per structure
    - SVG files are typically 10-100KB, matrices are ~1-10KB

TROUBLESHOOTING:
    - Download failures: Some UniProt IDs may not have AlphaFold structures
    - PDB files: PDB files require DSSP files which are not automatically generated
    - Alignment failures: Alignment requires foldseek to be installed and accessible
    - Memory errors: Very large proteins (>5000 residues) may cause memory issues

    Solutions:
    - Use --continue-on-error to skip failed structures
    - Use --skip-align if alignment is not needed
    - Limit structures with --max-structures for testing

REQUIREMENTS:
    - FlatProt installed
    - Structure files in input folder (.cif or .pdb files)
    - Optional: foldseek for alignment benchmarks
"""

import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def find_structure_files(
    structures_dir: Path, extensions: List[str] = None
) -> List[Path]:
    """
    Find all structure files in the given directory.

    Args:
        structures_dir: Directory containing structure files
        extensions: List of file extensions to look for (default: ['.cif', '.pdb', '.mmcif'])

    Returns:
        List of paths to structure files
    """
    if extensions is None:
        extensions = [".cif", ".pdb", ".mmcif"]

    structure_files = []

    if not structures_dir.exists():
        log.error(f"Structures directory not found: {structures_dir}")
        return structure_files

    for ext in extensions:
        pattern = f"*{ext}"
        files = list(structures_dir.glob(pattern))
        structure_files.extend(files)
        log.debug(f"Found {len(files)} {ext} files")

    # Remove duplicates and sort
    structure_files = sorted(set(structure_files))
    log.info(f"Found {len(structure_files)} total structure files in {structures_dir}")

    return structure_files


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size if file_path.exists() else 0


def get_protein_length(structure_file: Path) -> int:
    """Get protein length by parsing structure file."""
    try:
        from flatprot.io import GemmiStructureParser

        parser = GemmiStructureParser()
        structure = parser.parse_structure(structure_file)
        return len(structure.residues)
    except Exception as e:
        log.warning(f"Could not parse {structure_file.name} for length: {e}")
        return 0


def extract_protein_id(structure_file: Path) -> str:
    """Extract protein ID from structure filename."""
    # Handle AlphaFold naming: AF-P69905-F1-model_v4.cif -> P69905
    filename = structure_file.stem

    if filename.startswith("AF-") and "-F1-model" in filename:
        # AlphaFold format
        parts = filename.split("-")
        if len(parts) >= 2:
            return parts[1]

    # For other formats, just use the filename without extension
    return filename.split(".")[0]


def run_flatprot_project(
    structure_file: Path,
    output_file: Path,
    dssp_file: Optional[Path] = None,
    extra_args: List[str] = None,
) -> Tuple[float, bool, int, str]:
    """
    Run flatprot project command and measure runtime.

    Args:
        structure_file: Input structure file
        output_file: Output SVG file
        dssp_file: Optional DSSP file for PDB inputs
        extra_args: Additional command line arguments

    Returns:
        (runtime_seconds, success, output_file_size, error_message)
    """
    cmd = ["flatprot", "project", str(structure_file), "-o", str(output_file)]

    # Add DSSP file if provided and structure is PDB
    if dssp_file and dssp_file.exists():
        cmd.extend(["--dssp", str(dssp_file)])
    elif structure_file.suffix.lower() == ".pdb":
        # For PDB files without DSSP, we'll note this in the error
        log.warning(f"PDB file {structure_file.name} provided without DSSP file")

    if extra_args:
        cmd.extend(extra_args)

    start_time = time.perf_counter()
    try:
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True,
        )
        end_time = time.perf_counter()
        runtime = end_time - start_time
        output_size = get_file_size(output_file)
        return runtime, True, output_size, ""

    except subprocess.CalledProcessError as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        error_msg = f"Exit code {e.returncode}: {e.stderr.strip()}"
        log.error(f"FlatProt project failed for {structure_file.name}: {error_msg}")
        return runtime, False, 0, error_msg
    except subprocess.TimeoutExpired:
        error_msg = "Command timed out after 300 seconds"
        log.error(f"FlatProt project timed out for {structure_file.name}")
        return 300.0, False, 0, error_msg
    except Exception as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        error_msg = f"Unexpected error: {str(e)}"
        log.error(f"Unexpected error for {structure_file.name}: {error_msg}")
        return runtime, False, 0, error_msg


def run_flatprot_align_and_project(
    structure_file: Path,
    matrix_file: Path,
    svg_file: Path,
    extra_args: List[str] = None,
) -> Tuple[float, bool, int, int, str]:
    """
    Run flatprot align command followed by project command with the matrix, measuring combined runtime.

    Args:
        structure_file: Input structure file
        matrix_file: Output matrix file
        svg_file: Output SVG file for project command
        extra_args: Additional command line arguments

    Returns:
        (runtime_seconds, success, matrix_file_size, svg_file_size, error_message)
    """
    align_cmd = [
        "flatprot",
        "align",
        str(structure_file),
        "-m",
        str(matrix_file),
        "-p",
        "0.1",
    ]
    project_cmd = [
        "flatprot",
        "project",
        str(structure_file),
        "-o",
        str(svg_file),
        "--matrix",
        str(matrix_file),
    ]

    if extra_args:
        align_cmd.extend(extra_args)
        project_cmd.extend(extra_args)

    start_time = time.perf_counter()
    try:
        # Run align command
        _ = subprocess.run(
            align_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True,
        )

        # Run project command with matrix
        _ = subprocess.run(
            project_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True,
        )

        end_time = time.perf_counter()
        runtime = end_time - start_time
        matrix_size = get_file_size(matrix_file)
        svg_size = get_file_size(svg_file)
        return runtime, True, matrix_size, svg_size, ""

    except subprocess.CalledProcessError as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        error_msg = f"Exit code {e.returncode}: {e.stderr.strip()}"
        log.error(
            f"FlatProt align+project failed for {structure_file.name}: {error_msg}"
        )
        return runtime, False, 0, 0, error_msg
    except subprocess.TimeoutExpired:
        error_msg = "Command timed out after 300 seconds"
        log.error(f"FlatProt align+project timed out for {structure_file.name}")
        return 300.0, False, 0, 0, error_msg
    except Exception as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        error_msg = f"Unexpected error: {str(e)}"
        log.error(f"Unexpected error for {structure_file.name}: {error_msg}")
        return runtime, False, 0, 0, error_msg


def benchmark_structure(
    structure_file: Path,
    test_project: bool = True,
    test_align: bool = True,
) -> Dict:
    """
    Benchmark both project and align commands on a single structure.

    Args:
        structure_file: Path to structure file
        test_project: Whether to test project command
        test_align: Whether to test align command

    Returns:
        Dictionary with benchmark results
    """
    protein_id = extract_protein_id(structure_file)
    protein_length = get_protein_length(structure_file)
    structure_size = get_file_size(structure_file)

    log.info(f"Benchmarking {protein_id} ({structure_file.name})")
    log.info(f"  Length: {protein_length} residues, Size: {structure_size} bytes")

    result = {
        "protein_id": protein_id,
        "structure_file": structure_file.name,
        "structure_file_path": str(structure_file),
        "protein_length": protein_length,
        "structure_file_size": structure_size,
        "timestamp": datetime.now().isoformat(),
    }

    # Test project command
    if test_project:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
            svg_output = Path(tmp_file.name)
        (
            project_runtime,
            project_success,
            svg_size,
            project_error,
        ) = run_flatprot_project(structure_file, svg_output)

        result.update(
            {
                "project_runtime": project_runtime,
                "project_success": project_success,
                "svg_file_size": svg_size,
                "project_error": project_error,
            }
        )

        # Delete SVG file after measuring size
        if svg_output.exists():
            svg_output.unlink()
    else:
        result.update(
            {
                "project_runtime": 0.0,
                "project_success": False,
                "svg_file_size": 0,
                "project_error": "Skipped",
            }
        )

    # Test align command (align + project with matrix)
    if test_align:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_matrix:
            matrix_output = Path(tmp_matrix.name)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            align_svg_output = Path(tmp_svg.name)

        (
            align_runtime,
            align_success,
            matrix_size,
            align_svg_size,
            align_error,
        ) = run_flatprot_align_and_project(
            structure_file, matrix_output, align_svg_output
        )

        result.update(
            {
                "align_runtime": align_runtime,
                "align_success": align_success,
                "matrix_file_size": matrix_size,
                "align_svg_file_size": align_svg_size,
                "align_error": align_error,
            }
        )

        # Delete temporary files after measuring sizes
        if matrix_output.exists():
            matrix_output.unlink()
        if align_svg_output.exists():
            align_svg_output.unlink()
    else:
        result.update(
            {
                "align_runtime": 0.0,
                "align_success": False,
                "matrix_file_size": 0,
                "align_svg_file_size": 0,
                "align_error": "Skipped",
            }
        )

    return result


def save_results_tsv(results: List[Dict], output_file: Path):
    """Save benchmark results to TSV file."""
    if not results:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(results)

    log.info(f"Results saved to {output_file}")


def print_summary(results: List[Dict]):
    """Print benchmark summary statistics."""
    if not results:
        log.error("No results to summarize")
        return

    total_structures = len(results)
    successful_projects = sum(1 for r in results if r.get("project_success", False))
    successful_aligns = sum(1 for r in results if r.get("align_success", False))

    # Calculate average times (only for successful runs)
    project_times = [
        r["project_runtime"] for r in results if r.get("project_success", False)
    ]
    align_times = [r["align_runtime"] for r in results if r.get("align_success", False)]

    avg_project_time = sum(project_times) / len(project_times) if project_times else 0
    avg_align_time = sum(align_times) / len(align_times) if align_times else 0

    # Calculate file size statistics
    total_structure_size = sum(r["structure_file_size"] for r in results)
    total_svg_size = sum(
        r.get("svg_file_size", 0) for r in results if r.get("project_success", False)
    )
    total_align_svg_size = sum(
        r.get("align_svg_file_size", 0)
        for r in results
        if r.get("align_success", False)
    )
    total_matrix_size = sum(
        r.get("matrix_file_size", 0) for r in results if r.get("align_success", False)
    )

    # Length statistics
    lengths = [r["protein_length"] for r in results if r["protein_length"] > 0]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    log.info("=" * 60)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 60)
    log.info(f"Total structures processed: {total_structures}")
    log.info("")
    log.info("PROJECT COMMAND (inertia-based):")
    log.info(
        f"  Successful: {successful_projects}/{total_structures} ({successful_projects/total_structures*100:.1f}%)"
    )
    log.info(f"  Average time: {avg_project_time:.3f}s")
    log.info(f"  Total SVG size: {total_svg_size / (1024*1024):.1f} MB")
    log.info("")
    log.info("ALIGN + PROJECT COMMAND (with matrix):")
    log.info(
        f"  Successful: {successful_aligns}/{total_structures} ({successful_aligns/total_structures*100:.1f}%)"
    )
    log.info(f"  Average time: {avg_align_time:.3f}s")
    log.info(f"  Total matrix size: {total_matrix_size / (1024*1024):.1f} MB")
    log.info(f"  Total SVG size: {total_align_svg_size / (1024*1024):.1f} MB")
    log.info("")
    log.info("PROTEIN STATISTICS:")
    log.info(f"  Average length: {avg_length:.0f} residues")
    log.info(f"  Length range: {min_length} - {max_length} residues")
    log.info(f"  Total input size: {total_structure_size / (1024*1024):.1f} MB")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlatProt runtime performance on folder of structures"
    )
    parser.add_argument(
        "structures_dir", type=Path, help="Directory containing structure files"
    )
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=None,
        help="TSV file for results (default: runtime_results.tsv)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".cif", ".pdb", ".mmcif"],
        help="File extensions to process (default: .cif .pdb .mmcif)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to process (default: all)",
    )
    parser.add_argument(
        "--skip-project", action="store_true", help="Skip testing project command"
    )
    parser.add_argument(
        "--skip-align", action="store_true", help="Skip testing align command"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if individual structures fail",
    )

    args = parser.parse_args()

    # Validation
    if args.skip_project and args.skip_align:
        log.error("Cannot skip both project and align commands")
        return 1

    if args.results_tsv is None:
        args.results_tsv = Path("runtime_results.tsv")

    log.info("=" * 60)
    log.info("FLATPROT FOLDER BENCHMARK")
    log.info("=" * 60)
    log.info(f"Structures directory: {args.structures_dir}")
    log.info(f"Results file: {args.results_tsv}")
    log.info(f"Extensions: {args.extensions}")
    log.info(f"Test project: {not args.skip_project}")
    log.info(f"Test align: {not args.skip_align}")
    log.info("=" * 60)

    # Find structure files
    structure_files = find_structure_files(args.structures_dir, args.extensions)

    if not structure_files:
        log.error(f"No structure files found in {args.structures_dir}")
        return 1

    # Limit number of structures if requested
    if args.max_structures and args.max_structures < len(structure_files):
        structure_files = structure_files[: args.max_structures]
        log.info(f"Limited to first {args.max_structures} structures")

    log.info(f"Processing {len(structure_files)} structures...")

    # Run benchmarks
    results = []
    errors = []

    for i, structure_file in enumerate(structure_files, 1):
        log.info(f"[{i}/{len(structure_files)}] Processing {structure_file.name}")

        try:
            result = benchmark_structure(
                structure_file,
                test_project=not args.skip_project,
                test_align=not args.skip_align,
            )
            results.append(result)

            # Log progress for each command
            if not args.skip_project:
                project_status = "✓" if result["project_success"] else "✗"
                log.info(
                    f"  Project: {result['project_runtime']:.3f}s {project_status}"
                )

            if not args.skip_align:
                align_status = "✓" if result["align_success"] else "✗"
                log.info(f"  Align:   {result['align_runtime']:.3f}s {align_status}")

        except Exception as e:
            error_msg = f"Failed to benchmark {structure_file.name}: {e}"
            log.error(error_msg)
            errors.append(error_msg)

            if not args.continue_on_error:
                log.error("Stopping due to error (use --continue-on-error to continue)")
                break
            else:
                # Add a failed result entry
                failed_result = {
                    "protein_id": extract_protein_id(structure_file),
                    "structure_file": structure_file.name,
                    "structure_file_path": str(structure_file),
                    "protein_length": 0,
                    "structure_file_size": get_file_size(structure_file),
                    "project_runtime": 0.0,
                    "project_success": False,
                    "svg_file_size": 0,
                    "project_error": str(e),
                    "align_runtime": 0.0,
                    "align_success": False,
                    "matrix_file_size": 0,
                    "align_error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(failed_result)

    # Save results
    if results:
        save_results_tsv(results, args.results_tsv)
        print_summary(results)
    else:
        log.error("No results to save")
        return 1

    if errors:
        log.warning(f"Encountered {len(errors)} errors during processing")

    return 0


if __name__ == "__main__":
    sys.exit(main())
