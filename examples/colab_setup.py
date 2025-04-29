# examples/colab_setup.py
import os
import sys
import subprocess
from pathlib import Path


# Define run_cmd at the module level so it can be imported
def run_cmd(cmd_list, check=True, capture=False, env=None, shell=False):
    """Helper function to run subprocess commands and print status."""
    print(f" $ {cmd_list if isinstance(cmd_list, str) else ' '.join(cmd_list)}")
    try:
        result = subprocess.run(
            cmd_list,
            check=check,
            capture_output=capture,
            text=True,
            env=env,
            shell=shell,
        )
        # If capturing output, print some of it
        if capture and result.stdout:
            # Limit output length to avoid flooding
            output_str = result.stdout[:1000]
            ellipsis = "..." if len(result.stdout) > 1000 else ""
            print(f" -> Output (first 1000 chars):\n{output_str}{ellipsis}")
        # Always print stderr if it exists (often contains warnings or errors)
        if result.stderr:
            stderr_str = result.stderr  # No need to slice stderr
            print(f" -> Stderr:\n{stderr_str}", file=sys.stderr)
        # Raise an exception if check=True and the command failed
        result.check_returncode()
        print(" -> Command finished successfully.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        # Error details are usually in stderr, which should have been printed above
        raise  # Re-raise the exception to signal failure
    except FileNotFoundError:
        cmd_name = cmd_list[0] if cmd_list else "(empty command)"
        print(
            f"[ERROR] Command not found: {cmd_name}. Is it installed and in PATH?",
            file=sys.stderr,
        )
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error running command: {e}", file=sys.stderr)
        raise  # Re-raise the exception


def setup_colab_environment():
    """Installs dependencies and downloads data necessary for FlatProt examples in Colab."""
    print("--- Starting Google Colab Environment Setup ---")
    COLAB_BASE_DIR = Path(".")  # Assumes running in /content
    REPO_DIR_NAME = "FlatProt-main"  # Used for extracting data path

    # --- 1. Install FlatProt ---
    print("\n[1/4] Installing FlatProt via pip...")
    run_cmd(["uv", "tool", "install", "flatprot"])
    run_cmd(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "flatprot"]
    )
    # Verify installation (optional, but good practice)
    # run_cmd([sys.executable, "-m", "pip", "show", "flatprot"])
    print("[DONE] FlatProt installation step.")

    # --- 2. Install Foldseek ---
    print("\n[2/4] Installing Foldseek...")
    foldseek_url = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
    foldseek_tar = COLAB_BASE_DIR / "foldseek-linux-avx2.tar.gz"
    foldseek_dir = COLAB_BASE_DIR / "foldseek"
    foldseek_executable = foldseek_dir / "bin" / "foldseek"

    if not foldseek_executable.exists():
        print(f"Foldseek not found at {foldseek_executable}, downloading...")
        run_cmd(["wget", "-q", foldseek_url, "-O", str(foldseek_tar)])
        print("Extracting Foldseek...")
        # Ensure the target directory exists and is empty if necessary
        if foldseek_dir.exists():
            print(f"Removing existing Foldseek directory: {foldseek_dir}")
            run_cmd(["rm", "-rf", str(foldseek_dir)])
        foldseek_dir.mkdir(parents=True, exist_ok=True)
        # Extract directly into the foldseek directory
        run_cmd(["tar", "-xzf", str(foldseek_tar), "-C", str(COLAB_BASE_DIR)])
        foldseek_bin_path = str(foldseek_dir / "bin")
        # Add to PATH if not already there
        if foldseek_bin_path not in os.environ["PATH"]:
            os.environ["PATH"] = f"{foldseek_bin_path}:{os.environ['PATH']}"
            print(f"Added {foldseek_bin_path} to PATH")
        else:
            print(f"{foldseek_bin_path} already in PATH")
        # Clean up tarball
        print(f"Cleaning up {foldseek_tar}...")
        foldseek_tar.unlink(missing_ok=True)
    else:
        print(f"Foldseek already found at {foldseek_executable}.")
        # Ensure its bin dir is in PATH
        foldseek_bin_path = str(foldseek_dir / "bin")
        if foldseek_bin_path not in os.environ["PATH"]:
            os.environ["PATH"] = f"{foldseek_bin_path}:{os.environ['PATH']}"
            print(f"Added existing {foldseek_bin_path} to PATH")

    print("Verifying Foldseek installation by checking version...")
    run_cmd([str(foldseek_executable), "--help"], capture=True)
    print("[DONE] Foldseek installation step.")

    # --- 3. Install DSSP ---
    print("\n[3/4] Installing DSSP (mkdssp)...")
    # Check if mkdssp command exists using shutil.which
    import shutil

    if shutil.which("mkdssp") is None:
        print("'mkdssp' command not found, installing DSSP package...")
        print("Updating apt package list (this may take a moment)...")
        run_cmd(["sudo", "apt-get", "update", "-qq"])
        print("Installing DSSP package...")
        # Use the command that worked directly in the shell
        run_cmd(
            "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dssp",
            shell=True,
        )
    else:
        print("DSSP (mkdssp) command already found in PATH.")
    print("Verifying DSSP installation by checking version...")
    run_cmd(["mkdssp", "--version"], capture=True)
    print("[DONE] DSSP installation step.")

    # --- 4. Download Repository Data (if needed) ---
    print("\n[4/4] Ensuring 'data/' directory exists...")
    data_dir_local = COLAB_BASE_DIR / "data"
    if not data_dir_local.exists() or not any(data_dir_local.iterdir()):
        print("'data/' directory missing or empty, downloading from repository...")
        repo_zip_url = "https://github.com/t03i/FlatProt/archive/refs/heads/main.zip"
        repo_zip_file = COLAB_BASE_DIR / "repo.zip"
        repo_temp_dir = COLAB_BASE_DIR / "repo_temp"

        print(f"Downloading repository archive from {repo_zip_url}...")
        run_cmd(
            ["wget", "-nv", repo_zip_url, "-O", str(repo_zip_file)]
        )  # -nv for less verbose wget
        print(f"Extracting archive to {repo_temp_dir}...")
        # Ensure temp dir is clean/exists
        if repo_temp_dir.exists():
            run_cmd(["rm", "-rf", str(repo_temp_dir)])
        repo_temp_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(["unzip", "-o", "-q", str(repo_zip_file), "-d", str(repo_temp_dir)])

        extracted_repo_path = repo_temp_dir / REPO_DIR_NAME
        if extracted_repo_path.is_dir():
            source_data_path = extracted_repo_path / "data"
            if source_data_path.exists() and source_data_path.is_dir():
                print(
                    f"Moving data/ directory from {extracted_repo_path} to {data_dir_local}..."
                )
                # Ensure target data dir exists
                data_dir_local.mkdir(exist_ok=True)
                # Move contents using mv -T to handle potential existing target
                run_cmd(["mv", "-T", str(source_data_path), str(data_dir_local)])
                print("Moved data/ directory successfully.")
            else:
                print(
                    f"[WARN] 'data/' directory not found within extracted archive at {source_data_path}.",
                    file=sys.stderr,
                )
                data_dir_local.mkdir(exist_ok=True)  # Create empty data dir anyway
        else:
            print(
                f"[ERROR] Expected directory '{extracted_repo_path}' not found after extraction.",
                file=sys.stderr,
            )
            data_dir_local.mkdir(exist_ok=True)  # Create empty data dir

        print("Cleaning up downloaded archive files...")
        run_cmd(["rm", "-rf", str(repo_temp_dir), str(repo_zip_file)])
    else:
        print("'data/' directory already exists and is not empty. Skipping download.")

    # Final check for data directory
    if data_dir_local.exists():
        print(f"[DONE] 'data/' directory is present at {data_dir_local.resolve()}.")
    else:
        print(
            "[WARN] 'data/' directory is still missing after setup attempt.",
            file=sys.stderr,
        )

    print("\n--- Google Colab Environment Setup Finished ---")


# Note: No `if __name__ == "__main__":` block,
# this script is intended to be imported and the function called explicitly.
