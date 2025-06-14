#!/usr/bin/env -S uv --quiet run --script

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts",
# ]
# ///
"""
Generate single structure visualizations using multiple tools.

This script creates visualizations using ChimeraX, PyMOL, SSDraw, and Pro-origami
for any protein structure file.
"""

import glob
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from cyclopts import App

app = App()


def find_chimerax_executable() -> str:
    """Find the ChimeraX executable, especially on macOS.

    Returns:
        Path to ChimeraX executable.
    """
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
    """Get the directory of the currently running script.

    Returns:
        Path to the script directory.
    """
    return Path(__file__).parent.resolve()


def run_command(cmd: List[str]) -> Tuple[bool, str]:
    """Execute a command and return success status and message.

    Parameters
    ----------
    cmd
        The command to execute, as a list of strings.

    Returns
    -------
    A tuple containing:
        - Success status (bool).
        - Error message if failed, empty string if successful (str).
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stdout.strip()
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


def run_chimerax_visualization(
    structure_file: Path, output_dir: Path
) -> Tuple[bool, str]:
    """Run ChimeraX visualization for the structure.

    Parameters
    ----------
    structure_file
        Path to the input structure file.
    output_dir
        Directory to save output files.

    Returns
    -------
    Tuple of (success, message).
    """
    output_file = output_dir / f"chimerax_{structure_file.stem}.png"
    script_path = get_script_dir() / "tool_benchmark" / "chimerax_single.py"

    if not script_path.exists():
        return False, f"ChimeraX script not found: {script_path}"

    chimerax_executable = find_chimerax_executable()
    cmd = [
        chimerax_executable,
        "--script",
        f"{str(script_path)} {str(structure_file)} {str(output_file)}",
    ]

    print(f"Running ChimeraX: {' '.join(shlex.quote(arg) for arg in cmd)}")
    success, error_msg = run_command(cmd)

    if success and output_file.exists():
        return True, f"ChimeraX visualization saved to {output_file}"
    else:
        return False, f"ChimeraX failed: {error_msg}"


def run_pymol_visualization(structure_file: Path, output_dir: Path) -> Tuple[bool, str]:
    """Run PyMOL visualization for the structure.

    Parameters
    ----------
    structure_file
        Path to the input structure file.
    output_dir
        Directory to save output files.

    Returns
    -------
    Tuple of (success, message).
    """
    output_file = output_dir / f"pymol_{structure_file.stem}.png"
    script_path = get_script_dir() / "tool_benchmark" / "pymol_single.py"

    if not script_path.exists():
        return False, f"PyMOL script not found: {script_path}"

    cmd = [
        "pymol",
        "-cr",
        str(script_path),
        "--",
        str(structure_file),
        str(output_file),
    ]

    print(f"Running PyMOL: {' '.join(shlex.quote(arg) for arg in cmd)}")
    success, error_msg = run_command(cmd)

    if success and output_file.exists():
        return True, f"PyMOL visualization saved to {output_file}"
    else:
        return False, f"PyMOL failed: {error_msg}"


def setup_ssdraw() -> bool:
    """Setup SSDraw in the benchmark directory.

    Returns:
        True if setup successful, False otherwise.
    """
    benchmark_dir = get_script_dir() / "tool_benchmark"
    ssdraw_dir = benchmark_dir / "SSDraw"
    ssdraw_script = ssdraw_dir / "SSDraw" / "SSDraw.py"

    if not ssdraw_script.exists():
        print("Setting up SSDraw...")

        # Clone SSDraw to benchmark directory
        clone_cmd = [
            "git",
            "clone",
            "https://github.com/ncbi/SSDraw.git",
            str(ssdraw_dir),
        ]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to setup SSDraw: {result.stderr}")
            return False

        print(f"SSDraw setup complete: {ssdraw_script}")
    else:
        print(f"Using existing SSDraw setup: {ssdraw_script}")

    return True


def run_ssdraw_visualization(
    structure_file: Path, output_dir: Path
) -> Tuple[bool, str]:
    """Run SSDraw visualization for the structure.

    Parameters
    ----------
    structure_file
        Path to the input structure file.
    output_dir
        Directory to save output files.

    Returns
    -------
    Tuple of (success, message).
    """
    output_file = output_dir / f"ssdraw_{structure_file.stem}.png"
    script_path = get_script_dir() / "tool_benchmark" / "ssdraw_single.py"

    if not script_path.exists():
        return False, f"SSDraw script not found: {script_path}"

    # Setup SSDraw if needed
    if not setup_ssdraw():
        return False, "Failed to setup SSDraw"

    # Get SSDraw path from benchmark directory
    benchmark_dir = get_script_dir() / "tool_benchmark"
    ssdraw_script = benchmark_dir / "SSDraw" / "SSDraw" / "SSDraw.py"

    cmd = [
        "uv",
        "run",
        str(script_path),
        str(structure_file),
        str(output_file),
        "--ssdraw-path",
        str(ssdraw_script),
    ]

    print(f"Running SSDraw: {' '.join(shlex.quote(arg) for arg in cmd)}")
    success, error_msg = run_command(cmd)

    if success and output_file.exists():
        return True, f"SSDraw visualization saved to {output_file}"
    else:
        return False, f"SSDraw failed: {error_msg}"


def setup_proorigami() -> bool:
    """Setup pro-origami Docker image.

    Returns:
        True if setup successful, False otherwise.
    """
    benchmark_dir = get_script_dir() / "tool_benchmark"
    dockerfile_path = benchmark_dir / "Dockerfile.proorigami"

    # Check if Docker is available
    try:
        docker_check = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True
        )
        if docker_check.returncode != 0:
            print("Docker not available. pro-origami requires Docker to run.")
            return False
    except FileNotFoundError:
        print("Docker not installed. pro-origami requires Docker to run.")
        return False

    # Check if Docker image exists
    image_name = "proorigami:latest"
    image_check = subprocess.run(
        ["docker", "images", "-q", image_name], capture_output=True, text=True
    )

    if not image_check.stdout.strip():
        print("Building pro-origami Docker image...")

        if not dockerfile_path.exists():
            print(f"Dockerfile not found: {dockerfile_path}")
            return False

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
            print(f"Failed to build pro-origami Docker image: {result.stderr}")
            return False

        print("pro-origami Docker image built successfully")
    else:
        print("Using existing pro-origami Docker image")

    return True


def run_proorigami_visualization(
    structure_file: Path, output_dir: Path
) -> Tuple[bool, str]:
    """Run Pro-origami visualization for the structure.

    Parameters
    ----------
    structure_file
        Path to the input structure file.
    output_dir
        Directory to save output files.

    Returns
    -------
    Tuple of (success, message).
    """
    output_file = output_dir / f"proorigami_{structure_file.stem}.png"
    script_path = get_script_dir() / "tool_benchmark" / "proorigami_single.py"

    if not script_path.exists():
        return False, f"Pro-origami script not found: {script_path}"

    # Setup Pro-origami if needed
    if not setup_proorigami():
        return False, "Failed to setup Pro-origami"

    cmd = [
        "uv",
        "run",
        str(script_path),
        str(structure_file),
        str(output_file),
    ]

    print(f"Running Pro-origami: {' '.join(shlex.quote(arg) for arg in cmd)}")
    success, error_msg = run_command(cmd)

    if success and output_file.exists():
        return True, f"Pro-origami visualization saved to {output_file}"
    else:
        return False, f"Pro-origami failed: {error_msg}"


@app.default
def main(
    structure_file: Path,
    output_dir: Path,
    tools: Optional[str] = None,
) -> None:
    """Generate single structure visualizations using multiple tools.

    Parameters
    ----------
    structure_file
        Path to the input structure file (e.g., .cif, .pdb).
    output_dir
        Directory to save output files.
    tools
        Comma-separated list of tools to use. Options: chimerax,pymol,ssdraw,proorigami.
        Defaults to all tools.
    """
    # Check if structure file exists
    if not structure_file.exists():
        print(f"Error: Structure file not found: {structure_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default tools
    if tools is None:
        tools_to_run = ["chimerax", "pymol", "ssdraw", "proorigami"]
    else:
        tools_to_run = [tool.strip().lower() for tool in tools.split(",")]

    print(f"Generating visualizations for: {structure_file}")
    print(f"Output directory: {output_dir}")
    print(f"Tools to run: {', '.join(tools_to_run)}")
    print("=" * 60)

    results = []

    # Run each tool
    for tool in tools_to_run:
        print(f"\n--- Running {tool.upper()} ---")

        if tool == "chimerax":
            success, message = run_chimerax_visualization(structure_file, output_dir)
        elif tool == "pymol":
            success, message = run_pymol_visualization(structure_file, output_dir)
        elif tool == "ssdraw":
            success, message = run_ssdraw_visualization(structure_file, output_dir)
        elif tool == "proorigami":
            success, message = run_proorigami_visualization(structure_file, output_dir)
        else:
            print(f"Unknown tool: {tool}")
            continue

        results.append(
            {
                "tool": tool,
                "success": success,
                "message": message,
            }
        )

        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {message}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful_tools = [r for r in results if r["success"]]
    failed_tools = [r for r in results if not r["success"]]

    print(
        f"Successfully generated: {len(successful_tools)}/{len(results)} visualizations"
    )

    if successful_tools:
        print("\nSuccessful visualizations:")
        for result in successful_tools:
            print(f"  ✓ {result['tool'].upper()}")

    if failed_tools:
        print("\nFailed visualizations:")
        for result in failed_tools:
            print(f"  ✗ {result['tool'].upper()}: {result['message']}")

    print(f"\nOutput files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    app()
