"""FlatProt overlay command for creating combined visualizations from multiple structures."""

import glob
import sys
from pathlib import Path
from typing import List, Optional

import cyclopts

from flatprot.core import logger
from flatprot.utils.overlay_utils import create_overlay, OverlayConfig
from .utils import set_logging_level, CommonParameters

# Import drawsvg at module level for easier testing
try:
    import drawsvg
except ImportError:
    drawsvg = None


app = cyclopts.App()


@app.default
def overlay(
    file_patterns: List[str],
    *,
    output: str = "overlay.png",
    family: Optional[str] = None,
    alignment_mode: str = "family-identity",
    style: Optional[Path] = None,
    canvas_width: int = 1000,
    canvas_height: int = 1000,
    min_probability: float = 0.5,
    dpi: int = 300,
    no_clustering: bool = False,
    common: CommonParameters | None = None,
) -> None:
    """Create overlay visualization from multiple protein structures.

    Combines multiple protein structures into a single overlay image with
    opacity scaling based on structural similarity clustering.

    Requirements:
        For PNG/PDF output, Cairo graphics library must be installed:
        - macOS: brew install cairo
        - Ubuntu: sudo apt-get install libcairo2-dev
        - Windows: Install Cairo binaries or use conda

    Args:
        file_patterns: Glob pattern(s) or space-separated file paths (e.g., "structures/*.cif" file1.cif).
        output: Output file path (format determined by extension: .svg, .png, .pdf)
        family: SCOP family ID for fixed family alignment (e.g., "3000114")
        alignment_mode: Alignment strategy ("family-identity" or "inertia")
        style: Custom style TOML file path
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        min_probability: Minimum alignment probability threshold
        dpi: DPI for raster output formats
        no_clustering: Disable automatic structure clustering
        common: Common CLI parameters (quiet/verbose)

    Examples:
        flatprot overlay "structures/*.cif" -o overlay.png
        flatprot overlay file1.cif file2.cif --family 3000114 -o result.pdf
        flatprot overlay "data/*.cif" --alignment-mode inertia --dpi 600
    """
    # Configure logging
    set_logging_level(common)

    try:
        # Resolve input files from pattern or explicit paths
        input_files = resolve_input_files(file_patterns)

        if len(input_files) < 2:
            logger.error(f"At least 2 input files required, found {len(input_files)}")
            sys.exit(1)

        if not (common and common.quiet):
            logger.info(f"Found {len(input_files)} input files")
            for f in input_files:
                logger.info(f"  - {f}")

        # Validate output format
        output_path = Path(output)
        output_format = output_path.suffix.lower().lstrip(".")
        if output_format not in ["svg", "png", "pdf"]:
            logger.error(f"Unsupported output format: {output_format}")
            logger.error("Supported formats: svg, png, pdf")
            sys.exit(1)

        # Check Cairo availability for raster formats
        if output_format in ["png", "pdf"]:
            if drawsvg is None:
                logger.error("drawsvg library not available")
                sys.exit(1)
            if not hasattr(drawsvg, "_cairo_available") or not drawsvg._cairo_available:
                logger.error("Cairo library not available for PNG/PDF output")
                logger.error(
                    "Install Cairo: brew install cairo (macOS) or sudo apt-get install libcairo2-dev (Ubuntu)"
                )
                sys.exit(1)

        # Configure overlay settings
        config = OverlayConfig(
            alignment_mode=alignment_mode,
            target_family=family,
            min_probability=min_probability,
            clustering_enabled=not no_clustering,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            style_file=style,
            output_format=output_format,
            dpi=dpi,
            quiet=bool(common and common.quiet),
        )

        # Create overlay
        result_path = create_overlay(input_files, output_path, config)

        if not (common and common.quiet):
            logger.info(f"Overlay created successfully: {result_path}")

    except KeyboardInterrupt:
        logger.error("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create overlay: {e}")
        if not (common and common.quiet):
            logger.exception("Full traceback:")
        sys.exit(1)


def resolve_input_files(file_patterns: List[str]) -> List[Path]:
    """Resolve input files from a list of glob patterns or explicit paths.

    Args:
        file_patterns: A list of file paths or glob patterns.

    Returns:
        A list of unique, sorted Path objects representing the resolved files.

    Raises:
        FileNotFoundError: If a pattern matches no files.
        ValueError: If no input patterns are provided.
    """
    if not file_patterns:
        raise ValueError("No input files or patterns provided.")

    all_files: set[Path] = set()
    for pattern in file_patterns:
        found_files = glob.glob(pattern)
        if not found_files:
            raise FileNotFoundError(f"No files found matching: '{pattern}'")

        for f_path in found_files:
            all_files.add(Path(f_path))

    return sorted(list(all_files))


if __name__ == "__main__":
    app()
