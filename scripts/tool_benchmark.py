# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts",
#   "polars",
#   "psutil",
# ]
# ///
"""Benchmark various protein visualization tools."""

import csv
import glob
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import polars as pl
import psutil
from cyclopts import App, Parameter

app = App()


def find_chimerax_executable() -> str:
    """Find the ChimeraX executable, especially on macOS."""
    # 1. Check if 'ChimeraX' is in the system's PATH
    chimerax_path = shutil.which("ChimeraX")
    if chimerax_path:
        return chimerax_path

    # 2. If on macOS, check standard application directories
    if sys.platform == "darwin":
        # Use glob to find any version of ChimeraX in /Applications
        # e.g., /Applications/ChimeraX-1.8.app/Contents/MacOS/ChimeraX
        search_pattern = "/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX"
        possible_paths = glob.glob(search_pattern)
        if possible_paths:
            # Sort to get the latest version if multiple are installed
            possible_paths.sort()
            return possible_paths[-1]

    # 3. Fallback to default, which might fail but is the last resort.
    return "ChimeraX"


def get_script_dir() -> Path:
    """Get the directory of the currently running script."""
    return Path(__file__).parent.resolve()


def measure_execution(cmd: List[str]) -> Tuple[float, float, int, str]:
    """Execute a command and measure its performance.
    Measures execution time and peak memory usage of a command.
    Parameters
    ----------
    cmd
        The command to execute, as a list of strings.
    Returns
    -------
    A tuple containing:
        - Execution time in seconds (float).
        - Peak memory usage in MB (float).
        - Exit code (int).
        - Error message (str).
    """
    start_time = time.monotonic()
    try:
        process = psutil.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak_memory_mb = 0
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            try:
                # RSS is the Resident Set Size
                mem_info = process.memory_info().rss / (1024 * 1024)  # MB
                if mem_info > peak_memory_mb:
                    peak_memory_mb = mem_info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.01)  # Polling interval
        stdout = process.communicate()[0]
        exit_code = process.wait()
    except FileNotFoundError:
        return 0.0, 0.0, 127, f"Command not found: {cmd[0]}"
    except Exception as e:
        return time.monotonic() - start_time, 0.0, 1, str(e)

    exec_time = time.monotonic() - start_time
    error_msg = stdout.strip() if exit_code != 0 else ""
    return exec_time, peak_memory_mb, exit_code, error_msg


def log_result(
    writer,
    tool: str,
    method: str,
    structure: str,
    iteration: int,
    exec_time: float,
    memory: float,
    exit_code: int,
    error_msg: str,
    family_available: str,
):
    """Write a benchmark result to the CSV file."""
    writer.writerow(
        [
            tool,
            method,
            structure,
            iteration,
            f"{exec_time:.4f}",
            f"{memory:.4f}",
            exit_code,
            error_msg,
            family_available,
        ]
    )


def flatprot_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running FlatProt on a single structure."""
    output_file = temp_dir / f"flatprot_single_{structure.name}.svg"
    return ["flatprot", "project", str(structure), "-o", str(output_file)]


def flatprot_aligned_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running FlatProt on a single structure with prior alignment."""
    output_file = temp_dir / f"flatprot_aligned_{structure.name}.svg"
    script_path = get_script_dir() / "tool_benchmark" / "flatprot_aligned.py"
    if not script_path.exists():
        raise FileNotFoundError(f"FlatProt-aligned script not found: {script_path}")
    return [
        "uv",
        "run",
        str(script_path),
        str(structure),
        str(output_file),
    ]


def flatprot_family_cmd(structures: List[Path], temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running FlatProt on a family of structures."""
    output_file = temp_dir / f"flatprot_overlay_{int(time.time())}.svg"
    if not structures:
        return None
    return [
        "flatprot",
        "overlay",
        *[str(s) for s in structures],
        "--quiet",
        "--family",
        "3000114",
        "--output",
        str(output_file),
    ]


def flatprot_family_large_cmd(
    structures: List[Path], temp_dir: Path
) -> Optional[List[str]]:
    """Return the command for running FlatProt on a large family of structures."""
    return flatprot_family_cmd(structures, temp_dir)


def pymol_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running PyMOL on a single structure."""
    output_file = temp_dir / f"pymol_single_{structure.name}.png"
    script_path = get_script_dir() / "tool_benchmark" / "pymol_single.py"
    if not script_path.exists():
        raise FileNotFoundError(f"PyMOL script not found: {script_path}")
    return [
        "pymol",
        "-cr",
        str(script_path),
        "--",
        str(structure),
        str(output_file),
    ]


def pymol_family_cmd(structures: List[Path], temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running PyMOL on a family of structures."""
    output_file = temp_dir / f"pymol_family_{int(time.time())}.png"
    script_path = get_script_dir() / "tool_benchmark" / "pymol_family.py"
    if not script_path.exists():
        raise FileNotFoundError(f"PyMOL script not found: {script_path}")
    if not structures:
        return None
    return [
        "pymol",
        "-cr",
        str(script_path),
        "--",
        str(output_file),
        *[str(s) for s in structures],
    ]


def pymol_family_large_cmd(
    structures: List[Path], temp_dir: Path
) -> Optional[List[str]]:
    """Return the command for running PyMOL on a large family of structures."""
    return pymol_family_cmd(structures, temp_dir)


def chimerax_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running ChimeraX on a single structure."""
    output_file = temp_dir / f"chimerax_single_{structure.name}.png"
    script_path = (
        get_script_dir() / "tool_benchmark" / "chimerax_single.py"
    ).relative_to(get_script_dir().parent)
    if not script_path.exists():
        raise FileNotFoundError(f"ChimeraX script not found: {script_path}")
    chimerax_executable = find_chimerax_executable()
    return [
        chimerax_executable,
        "--script",
        f"{str(script_path)} {str(structure)} {str(output_file)}",
    ]


def chimerax_family_cmd(structures: List[Path], temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running ChimeraX on a family of structures."""
    output_file = temp_dir / f"chimerax_family_{int(time.time())}.png"
    script_path = (
        get_script_dir() / "tool_benchmark" / "chimerax_family.py"
    ).relative_to(get_script_dir().parent)
    if not script_path.exists():
        raise FileNotFoundError(f"ChimeraX script not found: {script_path}")
    chimerax_executable = find_chimerax_executable()
    if not structures:
        return None
    return [
        chimerax_executable,
        "--script",
        f"{str(script_path)} {str(output_file)} {" ".join(map(str, structures))}",
    ]


def chimerax_family_large_cmd(
    structures: List[Path], temp_dir: Path
) -> Optional[List[str]]:
    """Return the command for running ChimeraX on a large family of structures."""
    return chimerax_family_cmd(structures, temp_dir)


def setup_ssdraw() -> Path:
    """Setup SSDraw in the benchmark directory. Returns path to SSDraw script."""
    benchmark_dir = get_script_dir() / "tool_benchmark"
    ssdraw_dir = benchmark_dir / "SSDraw"
    ssdraw_script = ssdraw_dir / "SSDraw" / "SSDraw.py"

    if not ssdraw_script.exists():
        print("Setting up SSDraw for benchmarking...")

        # Clone SSDraw to benchmark directory
        clone_cmd = [
            "git",
            "clone",
            "https://github.com/ncbi/SSDraw.git",
            str(ssdraw_dir),
        ]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to setup SSDraw: {result.stderr}")

        print(f"SSDraw setup complete: {ssdraw_script}")
    else:
        print(f"Using existing SSDraw setup: {ssdraw_script}")

    return ssdraw_script


def cleanup_ssdraw():
    """Remove SSDraw from the benchmark directory."""
    benchmark_dir = get_script_dir() / "tool_benchmark"
    ssdraw_dir = benchmark_dir / "SSDraw"

    if ssdraw_dir.exists():
        shutil.rmtree(ssdraw_dir)
        print(f"Cleaned up SSDraw: {ssdraw_dir}")


def setup_proorigami() -> Path:
    """Setup pro-origami Docker image for benchmarking. Returns path to benchmark directory."""
    benchmark_dir = get_script_dir() / "tool_benchmark"
    dockerfile_path = benchmark_dir / "Dockerfile.proorigami"

    # Check if Docker is available
    try:
        docker_check = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True
        )
        if docker_check.returncode != 0:
            raise RuntimeError(
                "Docker not available. pro-origami requires Docker to run."
            )
    except FileNotFoundError:
        raise RuntimeError("Docker not installed. pro-origami requires Docker to run.")

    # Check if Docker image exists
    image_name = "proorigami:latest"
    image_check = subprocess.run(
        ["docker", "images", "-q", image_name], capture_output=True, text=True
    )

    if not image_check.stdout.strip():
        print("Building pro-origami Docker image for benchmarking...")

        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        # Build Docker image
        build_cmd = [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "-t",
            image_name,
            str(benchmark_dir),
        ]

        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Docker build stdout: {result.stdout}")
            print(f"Docker build stderr: {result.stderr}")
            raise RuntimeError(
                f"Failed to build pro-origami Docker image: {result.stderr}"
            )

        print("pro-origami Docker image built successfully")
    else:
        print("Using existing pro-origami Docker image")

    return benchmark_dir


def cleanup_proorigami():
    """Remove pro-origami Docker image and files."""
    try:
        # Remove Docker image
        image_name = "proorigami:latest"

        # Check if image exists and remove it
        image_check = subprocess.run(
            ["docker", "images", "-q", image_name], capture_output=True, text=True
        )

        if image_check.stdout.strip():
            remove_cmd = ["docker", "rmi", image_name]
            result = subprocess.run(remove_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Cleaned up pro-origami Docker image: {image_name}")
            else:
                print(
                    f"Warning: Failed to remove Docker image {image_name}: {result.stderr}"
                )

        print("pro-origami cleanup complete")
    except FileNotFoundError:
        print("Docker not available - no pro-origami cleanup needed")


def ssdraw_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running SSDraw on a single structure."""
    output_file = temp_dir / f"ssdraw_single_{structure.name}.png"
    script_path = get_script_dir() / "tool_benchmark" / "ssdraw_single.py"
    if not script_path.exists():
        raise FileNotFoundError(f"SSDraw script not found: {script_path}")

    # Get SSDraw path from local benchmark directory
    benchmark_dir = get_script_dir() / "tool_benchmark"
    ssdraw_script = benchmark_dir / "SSDraw" / "SSDraw" / "SSDraw.py"

    return [
        "uv",
        "run",
        str(script_path),
        str(structure),
        str(output_file),
        "--ssdraw-path",
        str(ssdraw_script),  # Pass SSDraw path directly
    ]


def proorigami_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running pro-origami on a single structure."""
    output_file = temp_dir / f"proorigami_single_{structure.name}.png"
    script_path = get_script_dir() / "tool_benchmark" / "proorigami_single.py"
    if not script_path.exists():
        raise FileNotFoundError(f"pro-origami script not found: {script_path}")

    # For Docker-based pro-origami, we don't need to pass a specific path
    return [
        "uv",
        "run",
        str(script_path),
        str(structure),
        str(output_file),
    ]


def print_summary(results_file: Path):
    """Print a quick summary of the benchmark results using Polars."""
    if not results_file.exists():
        print(f"Results file not found: {results_file}", file=sys.stderr)
        return
    try:
        df = pl.read_csv(results_file)
        print("\nQuick Summary:")
        print("=" * 50)
        success_rates_df = df.group_by("tool", "method").agg(
            (pl.col("exit_code") == 0).mean().mul(100).round(2).alias("success_rate")
        )
        success_rates = {
            (row["tool"], row["method"]): row["success_rate"]
            for row in success_rates_df.to_dicts()
        }
        print("Success Rates (%):", success_rates)
        successful = df.filter(pl.col("exit_code") == 0)
        if not successful.is_empty():
            avg_times_df = successful.group_by("tool", "method").agg(
                pl.col("execution_time_seconds").mean().round(3).alias("avg_time")
            )
            avg_times = {
                (row["tool"], row["method"]): row["avg_time"]
                for row in avg_times_df.to_dicts()
            }
            print("Average Execution Times (seconds):", avg_times)
        family_support_df = (
            df.filter(pl.col("method") == "family")
            .group_by("tool")
            .agg(pl.col("family_available").first())
        )
        family_support = {
            row["tool"]: row["family_available"] for row in family_support_df.to_dicts()
        }
        print("Family Support:", family_support)
    except Exception as e:
        print(f"Could not generate summary: {e}", file=sys.stderr)


@app.default
def benchmark(
    small_structure_glob: Annotated[
        str, Parameter(help="Glob pattern for 'small' and 'family' structure files.")
    ],
    large_structure_glob: Annotated[
        str, Parameter(help="Glob pattern for 'family_large' structure files.")
    ],
    n_iterations: Annotated[
        int, Parameter(help="Number of iterations to run for each tool/method.")
    ] = 5,
    output_file: Annotated[
        Path, Parameter(help="Path to save the CSV results.")
    ] = Path("benchmark_results.csv"),
    print_commands: Annotated[
        bool, Parameter(help="Print the example commands for generating images.")
    ] = False,
):
    """Benchmark various protein visualization tools."""
    small_structure_files = [
        Path(p) for p in glob.glob(small_structure_glob, recursive=True)
    ]
    if not small_structure_files:
        print(
            f"Warning: No structure files found matching small glob: {small_structure_glob}",
            file=sys.stderr,
        )

    large_structure_files = [
        Path(p) for p in glob.glob(large_structure_glob, recursive=True)
    ]
    if not large_structure_files:
        print(
            f"Warning: No structure files found matching large glob: {large_structure_glob}",
            file=sys.stderr,
        )

    if not small_structure_files and not large_structure_files:
        print(
            "Error: No structure files were found for either glob pattern. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(small_structure_files)} files for small benchmarks.")
    print(f"Found {len(large_structure_files)} files for large benchmarks.")

    tools_methods = [
        ("flatprot", "single", flatprot_single_cmd),
        ("flatprot", "aligned", flatprot_aligned_cmd),
        ("flatprot", "family", flatprot_family_cmd),
        ("flatprot", "family_large", flatprot_family_large_cmd),
        ("pymol", "single", pymol_single_cmd),
        ("pymol", "family", pymol_family_cmd),
        ("pymol", "family_large", pymol_family_large_cmd),
        ("chimerax", "single", chimerax_single_cmd),
        ("chimerax", "family", chimerax_family_cmd),
        ("chimerax", "family_large", chimerax_family_large_cmd),
        ("ssdraw", "single", ssdraw_single_cmd),
        # ("proorigami", "single", proorigami_single_cmd),
    ]

    # Setup SSDraw if needed for benchmarking
    ssdraw_tools = [tool for tool, method, func in tools_methods if tool == "ssdraw"]
    if ssdraw_tools:
        try:
            setup_ssdraw()
        except Exception as e:
            print(f"Warning: Failed to setup SSDraw: {e}")
            # Remove SSDraw from tools_methods if setup failed
            tools_methods = [
                (tool, method, func)
                for tool, method, func in tools_methods
                if tool != "ssdraw"
            ]

    # Setup pro-origami if needed for benchmarking
    proorigami_tools = [
        tool for tool, method, func in tools_methods if tool == "proorigami"
    ]
    if proorigami_tools:
        try:
            setup_proorigami()
        except Exception as e:
            print(f"Warning: Failed to setup pro-origami: {e}")
            # Remove pro-origami from tools_methods if setup failed
            tools_methods = [
                (tool, method, func)
                for tool, method, func in tools_methods
                if tool != "proorigami"
            ]

    try:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "tool",
                        "method",
                        "structure_file",
                        "iteration",
                        "execution_time_seconds",
                        "memory_mb",
                        "exit_code",
                        "error_message",
                        "family_available",
                    ]
                )
                for tool, method, func in tools_methods:
                    print(f"Benchmarking {tool} ({method})...")

                    if method.endswith("_large"):
                        structure_files = large_structure_files
                    else:
                        structure_files = small_structure_files

                    if not structure_files:
                        print(
                            f"    Skipping {tool} ({method}) because no structure files were found."
                        )
                        continue

                    for iteration in range(1, n_iterations + 1):
                        print(f"  Iteration {iteration}/{n_iterations}")
                        if method in ("single", "aligned"):
                            for structure in structure_files:
                                print(f"    Processing {structure.name}")
                                cmd = func(structure, temp_dir)
                                if cmd:
                                    if print_commands:
                                        print(f"      # Command: {shlex.join(cmd)}")
                                    (
                                        exec_time,
                                        memory,
                                        exit_code,
                                        error_msg,
                                    ) = measure_execution(cmd)
                                    log_result(
                                        writer,
                                        tool,
                                        method,
                                        structure.name,
                                        iteration,
                                        exec_time,
                                        memory,
                                        exit_code,
                                        error_msg,
                                        "N/A",
                                    )
                        else:  # family and family_large
                            print(
                                f"    Processing {method} with {len(structure_files)} structures"
                            )
                            cmd_or_status = func(structure_files, temp_dir)
                            family_available = "error"
                            if cmd_or_status == "UNSUPPORTED":
                                (
                                    exec_time,
                                    memory,
                                    exit_code,
                                    error_msg,
                                ) = (0.0, 0.0, 2, f"{tool} {method} not supported")
                                family_available = "false"
                            elif cmd_or_status:
                                if print_commands:
                                    print(
                                        f"      # Command: {shlex.join(cmd_or_status)}"
                                    )
                                (
                                    exec_time,
                                    memory,
                                    exit_code,
                                    error_msg,
                                ) = measure_execution(cmd_or_status)
                                if exit_code == 0:
                                    family_available = "true"
                            else:
                                (exec_time, memory, exit_code, error_msg) = (
                                    0.0,
                                    0.0,
                                    1,
                                    f"Could not create command for {tool} {method}",
                                )
                            log_result(
                                writer,
                                tool,
                                method,
                                "multiple_structures",
                                iteration,
                                exec_time,
                                memory,
                                exit_code,
                                error_msg,
                                family_available,
                            )
    finally:
        # Cleanup SSDraw if it was setup for benchmarking
        if ssdraw_tools:
            cleanup_ssdraw()

        # Cleanup pro-origami if it was setup for benchmarking
        if proorigami_tools:
            cleanup_proorigami()

    print(f"\nBenchmark completed. Results saved to: {output_file}")
    print_summary(output_file)


if __name__ == "__main__":
    app()
