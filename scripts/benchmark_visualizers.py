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


def flatprot_family_cmd(structures: List[Path], temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running FlatProt on a family of structures."""
    output_file = temp_dir / f"flatprot_overlay_{int(time.time())}.png"
    return [
        "flatprot",
        "overlay",
        *[str(s) for s in structures],
        "-o",
        str(output_file),
    ]


def pymol_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running PyMOL on a single structure."""
    output_file = temp_dir / f"pymol_single_{structure.name}.png"
    script_path = get_script_dir() / "benchmark" / "pymol_single.py"
    if not script_path.exists():
        raise FileNotFoundError(f"PyMOL script not found: {script_path}")
    return ["pymol", "-cr", str(script_path), str(structure), str(output_file)]


def pymol_family_cmd(structures: List[Path], temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running PyMOL on a family of structures."""
    output_file = temp_dir / f"pymol_family_{int(time.time())}.png"
    script_path = get_script_dir() / "benchmark" / "pymol_family.py"
    if not script_path.exists():
        raise FileNotFoundError(f"PyMOL script not found: {script_path}")
    if not structures:
        return None
    return [
        "pymol",
        "-cr",
        str(script_path),
        str(output_file),
        *[str(s) for s in structures],
    ]


def chimerax_single_cmd(structure: Path, temp_dir: Path) -> Optional[List[str]]:
    """Return the command for running ChimeraX on a single structure."""
    output_file = temp_dir / f"chimerax_single_{structure.name}.png"
    script_path = (get_script_dir() / "benchmark" / "chimerax_single.py").relative_to(
        get_script_dir().parent
    )
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
    script_path = (get_script_dir() / "benchmark" / "chimerax_family.py").relative_to(
        get_script_dir().parent
    )
    if not script_path.exists():
        raise FileNotFoundError(f"ChimeraX script not found: {script_path}")
    chimerax_executable = find_chimerax_executable()
    return [
        chimerax_executable,
        "--script",
        f"{str(script_path)} {str(output_file)} {' '.join(map(str, structures))}",
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
    structure_pattern: Annotated[
        str, Parameter(help="Glob pattern for structure files (e.g., 'data/*/*.cif').")
    ],
    n_iterations: Annotated[
        int, Parameter(help="Number of iterations to run for each tool/method.")
    ] = 5,
    output_file: Annotated[
        Path, Parameter(help="Path to save the CSV results.")
    ] = Path("benchmark_results.csv"),
):
    """Benchmark various protein visualization tools."""
    structure_files = [Path(p) for p in glob.glob(structure_pattern, recursive=True)]
    if not structure_files:
        print(
            f"Error: No structure files found matching pattern: {structure_pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(structure_files)} structure files.")
    tools_methods = [
        # ("flatprot", "single", flatprot_single_cmd),
        # ("flatprot", "family", flatprot_family_cmd),
        # ("pymol", "single", pymol_single_cmd),
        # ("pymol", "family", pymol_family_cmd),
        ("chimerax", "single", chimerax_single_cmd),
        ("chimerax", "family", chimerax_family_cmd),
    ]
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
                for iteration in range(1, n_iterations + 1):
                    print(f"  Iteration {iteration}/{n_iterations}")
                    if method == "single":
                        for structure in structure_files:
                            print(f"    Processing {structure.name}")
                            cmd = func(structure, temp_dir)
                            if cmd:
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
                    else:  # family
                        print(
                            f"    Processing family with {len(structure_files)} structures"
                        )
                        cmd_or_status = func(structure_files, temp_dir)
                        family_available = "error"
                        if cmd_or_status == "UNSUPPORTED":
                            (
                                exec_time,
                                memory,
                                exit_code,
                                error_msg,
                            ) = (0.0, 0.0, 2, f"{tool} family not supported")
                            family_available = "false"
                        elif cmd_or_status:
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
    print(f"\nBenchmark completed. Results saved to: {output_file}")
    print_summary(output_file)


if __name__ == "__main__":
    app()
