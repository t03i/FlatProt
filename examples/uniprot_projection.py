# %% [markdown]
# <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
# <h1 style="color: #2c3e50;">FlatProt Projection from UniProt ID</h1>
# </div>
#
# <div style="padding: 15px; background-color: #f9f9f9; border-left: 5px solid #3498db; margin: 20px 0; border-radius: 5px;">
# <b>Goal:</b> This notebook demonstrates how to fetch a protein structure from the
# AlphaFold Database (AFDB) using a UniProt ID, run DSSP to determine
# secondary structure, and then generate a 2D projection using FlatProt.
# </div>
#
# <div style="padding: 15px; background-color: #f9f9f9; border-radius: 5px; margin-bottom: 20px;">
# <b>Workflow:</b>
# <ol style="margin-top: 10px;">
#   <li><b>Setup:</b> Define the target UniProt ID and paths for output files.</li>
#   <li><b>Download:</b> Fetch the predicted structure file (CIF format) from AFDB.</li>
#   <li><b>DSSP:</b> Run <code>mkdssp</code> on the downloaded structure file.</li>
#   <li><b>Projection:</b> Run <code>flatprot project</code> using the structure file and the DSSP output to generate an SVG visualization.</li>
#   <li><b>Display:</b> Show the generated SVG file.</li>
# </ol>
# </div>

# %% [markdown]
# <div style="background-color: #e9f7ef; padding: 15px; border-radius: 10px; border-left: 5px solid #27ae60; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Configuration</h2>
# <p>Enter the UniProt ID for the protein of interest in the box below. This will be used for the entire analysis.</p>
# </div>

# %%
# @title 📋 Enter UniProt ID {display-mode: "form"}
# Enter the UniProt ID for the protein you want to project
UNIPROT_ID = "P69905"  # @param {type:"string"}

# Display some common examples as a guide
print(f"Using UniProt ID: {UNIPROT_ID}")
print("\nCommon examples:")
print("• P69905 - Human Hemoglobin subunit alpha")
print("• P02144 - Human Myoglobin")
print("• P0DTD1 - SARS-CoV-2 Nsp3 protein")

# %% [markdown]
# <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #2980b9; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Environment Setup for Google Colab</h2>
# <p>The following cell sets up the required dependencies for this notebook.</p>
# </div>

# %%
# @title 🔧 Environment Setup {display-mode: "form"}
# @hidden
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import time
from IPython.display import display, HTML, clear_output

# Check if in Colab and run setup
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print("Running in Google Colab. Fetching and executing setup script...")
    setup_script_url = (
        "https://raw.githubusercontent.com/t03i/FlatProt/main/examples/colab_setup.py"
    )
    setup_script_local_path = Path("colab_setup.py")

    print(f"Downloading {setup_script_url} to {setup_script_local_path}...")
    subprocess.run(
        ["wget", "-q", "-O", str(setup_script_local_path), setup_script_url],
        check=True,
    )
    print("Download complete.")

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    if setup_script_local_path.exists():
        import colab_setup

        print("Running colab_setup.setup_colab_environment()...")
        colab_setup.setup_colab_environment()
        print("Colab setup script finished.")
    else:
        raise RuntimeError(
            f"Setup script {setup_script_local_path} not found after download."
        )

# Define base_dir (works for both Colab and local)
base_dir = Path(".")

# --- Path Definitions ---
print(f"Setting up directories...")
tmp_dir_base = base_dir / "tmp"
tmp_dir_base.mkdir(parents=True, exist_ok=True)  # Ensure base tmp exists

# %%
# @title 🧰 Helper Functions {display-mode: "form"}
# @hidden
# Essential Imports
try:
    from IPython.display import display, HTML, clear_output
except ImportError:
    print("[WARN] IPython not found. SVG display will not work.", file=sys.stderr)

    # Define dummy functions if IPython is not available
    def display(obj: object) -> None:
        """Dummy display function."""
        print(f"Cannot display object: {obj}")

    def HTML(data: str) -> str:
        """Dummy HTML function."""
        print(f"Cannot display HTML: {data}")
        return data


# Helper function to display styled headers
def display_header(
    title: str, step_num: Optional[int] = None, color: str = "#3498db"
) -> None:
    """
    Display a styled header with optional step number.

    Args:
        title: The header title text
        step_num: Optional step number to display
        color: The accent color for the header
    """
    step_text = f"STEP {step_num}: " if step_num is not None else ""
    html = f"""
    <div style="background-color: #f5f9fa; padding: 10px 15px; border-radius: 8px;
                border-left: 5px solid {color}; margin: 20px 0 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h2 style="color: #2c3e50; margin: 0; font-size: 1.5em;">{step_text}{title}</h2>
    </div>
    """
    display(HTML(html))


# Helper function for styled status messages
def status_message(message: str, status: str = "info") -> None:
    """
    Display a styled status message.

    Args:
        message: The message to display
        status: Status type (info, success, warning, error)
    """
    color_map = {
        "info": "#3498db",  # Blue
        "success": "#2ecc71",  # Green
        "warning": "#f39c12",  # Orange
        "error": "#e74c3c",  # Red
        "working": "#9b59b6",  # Purple
    }

    icon_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "working": "⏳",
    }

    color = color_map.get(status.lower(), "#3498db")
    icon = icon_map.get(status.lower(), "ℹ️")

    html = f"""
    <div style="color: {color}; margin: 5px 0; padding: 5px; font-family: monospace; font-size: 1.1em;">
        {icon} {message}
    </div>
    """
    display(HTML(html))


# Function to show a simple progress bar
def show_progress(percent: float, width: int = 60) -> None:
    """
    Display a colored progress bar.

    Args:
        percent: Progress percentage (0-100)
        width: Width of the progress bar in characters
    """
    filled_len = int(width * percent / 100)
    bar = (
        f'<div style="width: 100%; background-color: #f0f0f0; border-radius: 5px; height: 20px; margin: 10px 0;">'
        f'<div style="width: {percent}%; background-color: #3498db; height: 100%; border-radius: 5px; '
        f'text-align: center; color: white; line-height: 20px; font-size: 12px;">{percent:.1f}%</div></div>'
    )
    display(HTML(bar))


# %%
# @title 📝 Configure Paths {display-mode: "form"}
# @hidden
display_header("Configuration", 1, "#27ae60")

# Define script-specific directories using the base paths
tmp_dir = tmp_dir_base / f"uniprot_{UNIPROT_ID}_projection"
tmp_dir.mkdir(parents=True, exist_ok=True)  # Ensure specific tmp exists
status_message(f"Using temporary directory: {tmp_dir.resolve()}", "info")

# Define file paths within the temporary directory
# AFDB URL Format: https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.cif
# We'll try v4 first, which is the latest as of writing.
afdb_cif_filename = f"AF-{UNIPROT_ID}-F1-model_v4.cif"
afdb_url = f"https://alphafold.ebi.ac.uk/files/{afdb_cif_filename}"
input_cif_path = tmp_dir / afdb_cif_filename
dssp_output_path = tmp_dir / f"{UNIPROT_ID}.dssp"
svg_output_path = tmp_dir / f"{UNIPROT_ID}_projection.svg"

# Show configuration details in a nice format
config_html = f"""
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; font-family: monospace;">
  <div style="color: #2c3e50; font-weight: bold; margin-bottom: 10px; font-size: 1.1em;">Configuration Summary:</div>
  <table style="width: 100%; border-collapse: collapse;">
    <tr><td style="padding: 5px; color: #7f8c8d; width: 180px;">UniProt ID:</td><td>{UNIPROT_ID}</td></tr>
    <tr><td style="padding: 5px; color: #7f8c8d;">AFDB URL:</td><td>{afdb_url}</td></tr>
    <tr><td style="padding: 5px; color: #7f8c8d;">Output CIF:</td><td>{input_cif_path}</td></tr>
    <tr><td style="padding: 5px; color: #7f8c8d;">Output DSSP:</td><td>{dssp_output_path}</td></tr>
    <tr><td style="padding: 5px; color: #7f8c8d;">Output SVG:</td><td>{svg_output_path}</td></tr>
  </table>
</div>
"""
display(HTML(config_html))

status_message("Configuration complete!", "success")

# %% [markdown]
# <div style="background-color: #f9f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #8e44ad; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Step 1: Download Structure from AlphaFold DB</h2>
# <p>Fetching the predicted protein structure in CIF format from the AlphaFold Database.</p>
# </div>

# %%
# @title 📥 Download AlphaFold Structure {display-mode: "form"}
# @hidden
display_header("Downloading Structure from AlphaFold DB", 1, "#8e44ad")


def download_file(url: str, output_path: Path) -> bool:
    """
    Downloads a file from a URL to a specified path using wget.

    Args:
        url: The URL to download from.
        output_path: The local path to save the file.

    Returns:
        True if download was successful, False otherwise.
    """
    status_message(f"Attempting to download from: {url}", "working")

    try:
        # Show a progress animation during download
        for i in range(5):
            clear_output(wait=True)
            display_header("Downloading Structure from AlphaFold DB", 1, "#8e44ad")
            status_message(f"Downloading from: {url}", "working")
            show_progress(i * 20)
            if i < 4:  # Skip sleep on last iteration
                time.sleep(0.5)

        # Use wget for downloading. -q for quiet, -O for output file.
        cmd = ["wget", "-q", "-O", str(output_path), url]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Final progress
        clear_output(wait=True)
        display_header("Downloading Structure from AlphaFold DB", 1, "#8e44ad")
        status_message(f"Downloading from: {url}", "info")
        show_progress(100)
        status_message(f"Successfully downloaded to: {output_path}", "success")
        return True
    except subprocess.CalledProcessError as e:
        clear_output(wait=True)
        display_header("Downloading Structure from AlphaFold DB", 1, "#8e44ad")
        status_message(f"Failed to download {url}. Error code: {e.returncode}", "error")
        if e.stderr:
            error_html = f"""
            <div style="background-color: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; font-family: monospace;">
                <div style="color: #c0392b; font-weight: bold;">Error Output:</div>
                <pre style="margin: 5px 0; color: #7f8c8d;">{e.stderr.strip()}</pre>
            </div>
            """
            display(HTML(error_html))
        if e.stdout:
            display(HTML(f"<pre>{e.stdout.strip()}</pre>"))
        # Clean up partially downloaded file if it exists
        output_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        clear_output(wait=True)
        display_header("Downloading Structure from AlphaFold DB", 1, "#8e44ad")
        status_message(f"An unexpected error occurred during download: {e}", "error")
        output_path.unlink(missing_ok=True)
        return False


# Download the structure
if not input_cif_path.exists():
    download_successful = download_file(afdb_url, input_cif_path)
    if not download_successful:
        # Optional: Could add fallback logic here to try older AFDB versions if needed
        error_html = """
        <div style="background-color: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0;">
            <div style="color: #c0392b; font-weight: bold;">Critical Error:</div>
            <div style="margin-top: 5px;">Could not download structure. Cannot proceed further.</div>
        </div>
        """
        display(HTML(error_html))
        raise RuntimeError(f"Could not download structure for {UNIPROT_ID}. Exiting.")
else:
    status_message(f"Structure file already exists: {input_cif_path}", "success")


# %% [markdown]
# <div style="background-color: #fff7e6; padding: 15px; border-radius: 10px; border-left: 5px solid #f39c12; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Step 2: Run DSSP</h2>
# <p>Calculating secondary structure information from the downloaded CIF file using the <code>mkdssp</code> tool.</p>
# </div>

# %%
# @title 🧬 Run DSSP Analysis {display-mode: "form"}
# @hidden
display_header("Running DSSP", 2, "#f39c12")


def run_mkdssp(input_cif: Path, output_dssp: Path) -> bool:
    """
    Runs the mkdssp command on an input CIF file.

    Args:
        input_cif: Path to the input structure file (CIF format).
        output_dssp: Path to save the DSSP output file.

    Returns:
        True if DSSP ran successfully, False otherwise.
    """
    if not input_cif.exists():
        status_message(f"Input CIF file not found: {input_cif}", "error")
        return False

    cmd = ["mkdssp", "-i", str(input_cif), "-o", str(output_dssp)]

    # Display command
    cmd_html = f"""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; font-family: monospace;">
        <span style="color: #2c3e50;">Running command: </span>
        <span style="color: #2980b9;">{" ".join(cmd)}</span>
    </div>
    """
    display(HTML(cmd_html))

    # Show progress animation
    for i in range(5):
        show_progress(i * 20)
        if i < 4:  # Skip sleep on last iteration
            time.sleep(0.3)

    try:
        # Check if DSSP command exists first (optional but good practice)
        # We rely on colab_setup or user environment for DSSP installation
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Show final progress
        show_progress(100)
        status_message(f"Successfully generated DSSP file: {output_dssp}", "success")

        if process.stdout:  # DSSP might print info to stdout
            stdout_html = f"""
            <div style="background-color: #f8f9fa; border-left: 4px solid #2980b9; padding: 10px; margin: 10px 0; max-height: 200px; overflow-y: auto; font-family: monospace;">
                <div style="color: #2980b9; font-weight: bold;">DSSP Output:</div>
                <pre style="margin: 5px 0; color: #7f8c8d;">{process.stdout}</pre>
            </div>
            """
            display(HTML(stdout_html))
        return True
    except FileNotFoundError:
        status_message(
            "'mkdssp' command not found. Ensure DSSP is installed and in PATH.", "error"
        )
        if IN_COLAB:
            error_html = """
            <div style="background-color: #fff8e1; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0;">
                Check the 'colab_setup.py' script installation steps.
            </div>
            """
            display(HTML(error_html))
        return False
    except subprocess.CalledProcessError as e:
        status_message(f"mkdssp failed with code {e.returncode}", "error")
        if e.stderr:
            error_html = f"""
            <div style="background-color: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; font-family: monospace;">
                <div style="color: #c0392b; font-weight: bold;">Error Output:</div>
                <pre style="margin: 5px 0; color: #7f8c8d;">{e.stderr}</pre>
            </div>
            """
            display(HTML(error_html))
        if e.stdout:
            display(HTML(f"<pre>{e.stdout}</pre>"))
        return False
    except Exception as e:
        status_message(f"An unexpected error occurred running mkdssp: {e}", "error")
        return False


# Run DSSP
dssp_successful = run_mkdssp(input_cif_path, dssp_output_path)
if not dssp_successful:
    warning_html = """
    <div style="background-color: #fff8e1; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0;">
        <div style="color: #d35400; font-weight: bold;">Warning:</div>
        <div style="margin-top: 5px;">DSSP execution failed. Proceeding with FlatProt projection WITHOUT DSSP data.</div>
        <div style="margin-top: 5px; font-style: italic; color: #7f8c8d;">
            Note: Secondary structure coloring will not be available in the projection.
        </div>
    </div>
    """
    display(HTML(warning_html))
    # Set path to None to signal projection to skip it
    dssp_output_path = None


# %% [markdown]
# <div style="background-color: #e8f6f3; padding: 15px; border-radius: 10px; border-left: 5px solid #16a085; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Step 3: Run FlatProt Projection</h2>
# <p>Generating the 2D projection of the protein structure using <code>flatprot project</code>.</p>
# </div>

# %%
# @title 🎨 Generate FlatProt Projection {display-mode: "form"}
# @hidden
display_header("Running FlatProt Projection", 3, "#16a085")


def run_flatprot_project(
    input_path: Path,
    output_path: Path,
    dssp_path: Optional[Path] = None,
    canvas_args: str = "--canvas-width 500 --canvas-height 400",
) -> bool:
    """
    Runs the flatprot project command.

    Args:
        input_path: Path to the input structure file (CIF).
        output_path: Path to save the output SVG file.
        dssp_path: Optional path to the DSSP file.
        canvas_args: String containing canvas dimension arguments.

    Returns:
        True if projection was successful, False otherwise.
    """
    if not input_path.exists():
        status_message(f"Input structure file not found: {input_path}", "error")
        return False

    cmd = [
        "uv",
        "run",
        "flatprot",
        "project",
        str(input_path),
        "-o",
        str(output_path),
        "--quiet",  # Reduce verbosity
    ]
    if dssp_path and dssp_path.exists():
        cmd.extend(["--dssp", str(dssp_path)])
        status_message(f"Using DSSP file: {dssp_path}", "info")
    elif dssp_path:
        status_message(
            f"DSSP file specified but not found: {dssp_path}. Projecting without it.",
            "warning",
        )
    else:
        status_message("Projecting without DSSP file.", "info")

    # Add canvas arguments if provided
    if canvas_args:
        cmd.extend(canvas_args.split())

    # Display command
    cmd_html = f"""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; font-family: monospace;">
        <span style="color: #2c3e50;">Running command: </span>
        <span style="color: #16a085;">{" ".join(cmd)}</span>
    </div>
    """
    display(HTML(cmd_html))

    # Show progress animation for projection process
    status_message("Generating projection...", "working")
    for i in range(11):
        show_progress(i * 10)
        if i < 10:  # Skip sleep on last iteration
            time.sleep(0.2)

    try:
        # Prefer run_cmd from colab_setup if available and defined
        if "colab_setup" in sys.modules and hasattr(colab_setup, "run_cmd"):
            colab_setup.run_cmd(cmd)
        else:
            # Fallback to simple subprocess run
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        status_message(f"Successfully generated projection: {output_path}", "success")
        return True
    except subprocess.CalledProcessError as e:
        status_message(f"FlatProt projection failed with code {e.returncode}", "error")
        if e.stderr:
            error_html = f"""
            <div style="background-color: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; font-family: monospace;">
                <div style="color: #c0392b; font-weight: bold;">Error Output:</div>
                <pre style="margin: 5px 0; color: #7f8c8d;">{e.stderr.strip()}</pre>
            </div>
            """
            display(HTML(error_html))
        if e.stdout:
            display(HTML(f"<pre>{e.stdout.strip()}</pre>"))
        return False
    except Exception as e:
        status_message(f"An unexpected error occurred during projection: {e}", "error")
        return False


# Run the projection
projection_successful = run_flatprot_project(
    input_cif_path, svg_output_path, dssp_output_path
)

if not projection_successful:
    error_html = """
    <div style="background-color: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0;">
        <div style="color: #c0392b; font-weight: bold;">Critical Error:</div>
        <div style="margin-top: 5px;">FlatProt projection failed. Cannot display result.</div>
    </div>
    """
    display(HTML(error_html))
    raise RuntimeError("FlatProt projection failed. Cannot display result.")


# %% [markdown]
# <div style="background-color: #eaf2f8; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 20px;">
# <h2 style="color: #2c3e50; margin-top: 0;">Results: 2D Protein Structure Visualization</h2>
# <p>Displaying the generated 2D projection of the protein structure.</p>
# </div>

# %%
# @title 🖼️ Display Results {display-mode: "form"}
# @hidden
display_header("Displaying Generated SVG", 4, "#3498db")


def display_svg_files(
    svg_files: List[str | Path],
    titles: Optional[List[str]] = None,
    width: str = "90%",  # Default to wider for single image
) -> None:
    """
    Display one or more SVG files side by side in a Jupyter environment.

    Args:
        svg_files: A list of paths (as strings or Path objects) to the SVG files.
        titles: An optional list of titles for each SVG. If None or mismatched,
                generic titles or filenames will be used.
        width: The CSS width property for each SVG container (e.g., '30%', '400px').
               Adjust based on the number of SVGs.
    """
    if not svg_files:
        status_message("No SVG files provided to display.", "warning")
        return

    num_files = len(svg_files)
    if titles is None or len(titles) != num_files:
        status_message("Using filenames as titles for SVG display.", "info")
        titles = [Path(f).name for f in svg_files]

    # Enhanced HTML structure with better styling
    html = '<div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; width: 100%;">'

    for i, (svg_file_path, title) in enumerate(zip(svg_files, titles)):
        svg_path = Path(svg_file_path)
        container_style = (
            f"width: {width}; max-width: 800px; border: 1px solid #ddd; text-align: center; padding: 20px; "
            f"margin: 15px auto; border-radius: 12px; background-color: white; "
            f"box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: all 0.3s ease;"
        )

        if not svg_path.exists():
            status_message(
                f"SVG file not found: {svg_path}. Skipping display.", "warning"
            )
            html += f"""
            <div style="{container_style}">
                <h3 style="margin-bottom: 10px; font-family: sans-serif; color: #2c3e50;">{title}</h3>
                <div style="color: #e74c3c; padding: 20px; background-color: #ffebee; border-radius: 8px;">
                    <i>File not found</i>
                </div>
            </div>
            """
            continue

        try:
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()

            # Ensure SVG is responsive within its container
            svg_content = svg_content.replace(
                "<svg ",
                '<svg style="width: 100%; height: auto; display: block; margin: auto; transition: all 0.3s ease;" ',
                1,  # Replace only the first occurrence
            )

            # Add protein info
            protein_info = f"""
            <div style="margin-top: 15px; text-align: left; background-color: #f8f9fa; padding: 12px; border-radius: 8px;">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 5px;">Protein Information:</div>
                <div style="font-size: 0.9em; color: #7f8c8d;">
                    <span style="font-weight: bold;">UniProt ID:</span> {UNIPROT_ID}
                </div>
                <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 3px;">
                    <span style="font-weight: bold;">Source:</span> AlphaFold Database
                </div>
                <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 3px;">
                    <span style="font-weight: bold;">Secondary Structure:</span> {
                "Included (from DSSP)" if dssp_output_path else "Not included"
            }
                </div>
            </div>
            """

            html += f"""
            <div style="{container_style}">
                <h3 style="margin-bottom: 15px; font-family: sans-serif; color: #2c3e50; word-wrap: break-word;">{title}</h3>
                {svg_content}
                {protein_info}
                <div style="margin-top: 10px; font-size: 0.8em; color: #95a5a6; text-align: right;">
                    Generated with FlatProt
                </div>
            </div>
            """
        except Exception as e:
            status_message(f"Failed to read or process SVG {svg_path}: {e}", "error")
            html += f"""
             <div style="{container_style}">
                 <h3 style="margin-bottom: 10px; font-family: sans-serif; color: #2c3e50;">{title}</h3>
                 <div style="color: #e74c3c; padding: 20px; background-color: #ffebee; border-radius: 8px;">
                    <i>Error loading SVG: {e}</i>
                 </div>
             </div>
             """

    html += "</div>"
    # Ensure display is imported and works
    try:
        display(HTML(html))
    except NameError:
        status_message(
            "Cannot display HTML. 'display' function not available.", "error"
        )
        print("--- HTML Content ---")
        print(html)  # Print raw HTML as fallback


# Display the single generated SVG
if svg_output_path.exists():
    status_message("Displaying final projection...", "success")
    display_svg_files(
        svg_files=[svg_output_path],
        titles=[f"FlatProt Projection: {UNIPROT_ID}"],
        width="95%",  # Make single SVG display wide
    )
else:
    status_message(f"Final SVG file not found at {svg_output_path}", "error")


# Add a completion banner
completion_html = """
<div style="margin-top: 30px; padding: 15px; background-color: #e8f5e9; border-radius: 10px; text-align: center; border: 1px solid #81c784;">
    <h2 style="color: #2e7d32; margin-top: 0;">✅ Analysis Complete</h2>
    <p style="color: #2c3e50;">Your protein structure has been successfully processed and visualized.</p>
</div>
"""
display(HTML(completion_html))

# %% [markdown]
# <div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid #ddd;">
# <p style="color: #7f8c8d; font-size: 0.9em;">
#   FlatProt - Developed by the <a href="https://github.com/rostlab/FlatProt" style="color: #3498db; text-decoration: none;">Rostlab</a>
# </p>
# </div>
