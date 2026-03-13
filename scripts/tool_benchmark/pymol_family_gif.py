# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Create a GIF that progressively adds aligned protein structures.

Frames are built cumulatively: frame 1 shows one structure, frame 2 shows
the first two, and so on — demonstrating how the overlay grows from clean to
complex.  The camera is fixed across all frames so the viewer can focus on
the structural accumulation rather than camera motion.

Usage (via PyMOL's bundled Python interpreter, headless):
    pymol -cq pymol_family_gif.py -- output.gif structure1.cif structure2.cif ...

Optional flags (after the '--' separator):
    --hold-first INT   Extra frames to hold the single-structure opening shot.
                       Default: 3.
    --hold-last INT    Extra frames to hold the full-overlay closing shot.
                       Default: 5.
    --duration INT     Duration of each frame in milliseconds. Default: 600.
    --width INT        Rendered frame width in pixels. Default: 800.
    --height INT       Rendered frame height in pixels. Default: 600.
    --dpi INT          DPI for PyMOL rendering. Default: 150.
    --loop INT         GIF loop count; 0 = loop forever. Default: 0.
    --no-align         Skip alignment; use raw coordinates as loaded.
    --color-by-ss      Color by secondary structure instead of per-object colors.
    --opacity FLOAT    Cartoon opacity 0.0–1.0. Default: 1.0.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from pymol import cmd

_PALETTE: List[str] = [
    "cyan",
    "magenta",
    "yellow",
    "orange",
    "lime",
    "lightblue",
    "salmon",
    "violet",
    "wheat",
    "teal",
    "hotpink",
    "olive",
    "slate",
    "forest",
    "deepteal",
    "chocolate",
    "tv_blue",
    "tv_green",
    "tv_red",
    "tv_yellow",
]


def _apply_ss_colors() -> None:
    """Color visible objects by secondary structure (helices red, sheets blue)."""
    cmd.color("gray70", "all")
    cmd.color("red", "ss h")
    cmd.color("blue", "ss s")


def _render_frame(path: str, width: int, height: int, dpi: int) -> None:
    """Render the current PyMOL scene to a PNG file.

    Parameters
    ----------
    path
        Output file path (must end in .png).
    width
        Frame width in pixels.
    height
        Frame height in pixels.
    dpi
        Dots per inch for the render.
    """
    cmd.png(path, width=width, height=height, dpi=dpi, ray=1, quiet=1)


def main(
    output_gif: Path,
    structure_files: List[Path],
    hold_first: int = 0,
    hold_last: int = 10,
    duration: int = 300,
    width: int = 800,
    height: int = 600,
    dpi: int = 300,
    loop: int = 0,
    align: bool = True,
    color_by_ss: bool = False,
    opacity: float = 1.0,
) -> None:
    """Build a cumulative-overlay GIF from a set of aligned protein structures.

    Parameters
    ----------
    output_gif
        Path to write the output GIF.
    structure_files
        Ordered list of structure files; the first is the alignment target.
    hold_first
        Number of extra frames to hold the single-structure opening shot.
    hold_last
        Number of extra frames to hold the fully-stacked closing shot.
    duration
        Per-frame duration in milliseconds.
    width
        Rendered image width in pixels.
    height
        Rendered image height in pixels.
    dpi
        DPI used by PyMOL when ray-tracing each frame.
    loop
        GIF loop count (0 = infinite).
    align
        If True, align every structure onto the first using PyMOL's align.
    color_by_ss
        If True, use secondary-structure coloring instead of per-object colors.
    opacity
        Cartoon opacity [0.0, 1.0].
    """
    if not structure_files:
        print("Error: no structure files provided.", file=sys.stderr)
        cmd.quit(1)

    missing = [f for f in structure_files if not f.exists()]
    if missing:
        for f in missing:
            print(f"Error: file not found: {f}", file=sys.stderr)
        cmd.quit(1)

    # ── Load all structures ───────────────────────────────────────────────────
    for sf in structure_files:
        cmd.load(str(sf))
        print(f"Loaded: {sf.name}")

    object_names: List[str] = cmd.get_names("objects")
    n = len(object_names)

    # ── Align everything to the first structure ───────────────────────────────
    if align and n > 1:
        target = object_names[0]
        for mobile in object_names[1:]:
            result = cmd.align(mobile, target)
            rmsd = result[0] if isinstance(result, (list, tuple)) else "?"
            print(f"  aligned {mobile} → {target}  RMSD = {rmsd:.3f} Å")

    # ── Global style settings ─────────────────────────────────────────────────
    cmd.do("as cartoon")
    cmd.bg_color("white")
    if opacity < 1.0:
        cmd.set("cartoon_transparency", 1.0 - opacity, "all")

    # ── Fix camera on the full ensemble ──────────────────────────────────────
    # Orient with all objects visible so the view encompasses every structure.
    cmd.show("cartoon", "all")
    cmd.orient("all")
    cmd.zoom("all", buffer=5)
    fixed_view = cmd.get_view()

    # Hide everything; we'll reveal objects one by one per frame.
    cmd.hide("everything", "all")

    # ── Render frames ─────────────────────────────────────────────────────────
    convert_bin = shutil.which("magick") or shutil.which("convert")
    if convert_bin is None:
        print(
            "Error: ImageMagick not found.  Install it with:\n"
            "  brew install imagemagick",
            file=sys.stderr,
        )
        cmd.quit(1)

    # ImageMagick -delay is in centiseconds (1/100 s).
    delay_cs = max(1, duration // 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build the ordered list of frame file paths that will be passed to
        # ImageMagick, duplicating paths to implement per-step holds.
        frame_paths: List[str] = []

        for step in range(1, n + 1):
            # Show the next object and color it.
            new_obj = object_names[step - 1]
            cmd.show("cartoon", new_obj)

            if color_by_ss:
                _apply_ss_colors()
            else:
                for idx in range(step):
                    cmd.color(_PALETTE[idx % len(_PALETTE)], object_names[idx])

            # Restore the fixed camera before rendering.
            cmd.set_view(fixed_view)

            frame_path = str(Path(tmpdir) / f"frame_{step:04d}.png")
            print(f"  rendering frame {step}/{n}: {new_obj} …")
            _render_frame(frame_path, width, height, dpi)

            # Repeat the path to hold the frame for the requested duration.
            repeats = hold_first + 1 if step == 1 else 1
            repeats = hold_last + 1 if step == n else repeats
            frame_paths.extend([frame_path] * repeats)

        # ── Assemble GIF via ImageMagick ──────────────────────────────────────
        output_gif.parent.mkdir(parents=True, exist_ok=True)

        cmd_parts = (
            [convert_bin, "-delay", str(delay_cs), "-loop", str(loop)]
            + frame_paths
            + [str(output_gif)]
        )
        print("Assembling GIF …")
        subprocess.run(cmd_parts, check=True)

    total_frames = len(frame_paths)
    total_sec = total_frames * duration / 1000
    print(
        f"\nGIF saved to: {output_gif}\n"
        f"  {total_frames} frames  ·  {duration} ms/frame  ·  ~{total_sec:.1f}s total"
    )


# PyMOL exec()s scripts rather than running them as __main__, so we cannot
# guard with `if __name__ == "__main__"`.  Run unconditionally at import time.
argv = sys.argv[1:]

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("output_gif", type=Path, help="Path to write the output GIF.")
parser.add_argument(
    "structure_files",
    type=Path,
    nargs="+",
    help="Structure files in the order they should be added (CIF, PDB, …).",
)
parser.add_argument(
    "--hold-first",
    type=int,
    default=1,
    metavar="INT",
    help="Extra frames to hold the opening single-structure shot (default: 3).",
)
parser.add_argument(
    "--hold-last",
    type=int,
    default=10,
    metavar="INT",
    help="Extra frames to hold the final full-overlay shot (default: 5).",
)
parser.add_argument(
    "--duration",
    type=int,
    default=450,
    metavar="MS",
    help="Duration of each frame in milliseconds (default: 600).",
)
parser.add_argument(
    "--width",
    type=int,
    default=800,
    metavar="PX",
    help="Frame width in pixels (default: 800).",
)
parser.add_argument(
    "--height",
    type=int,
    default=600,
    metavar="PX",
    help="Frame height in pixels (default: 600).",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=300,
    metavar="INT",
    help="DPI for PyMOL rendering (default: 150).",
)
parser.add_argument(
    "--loop",
    type=int,
    default=0,
    metavar="INT",
    help="GIF loop count; 0 = loop forever (default: 0).",
)
parser.add_argument(
    "--no-align",
    action="store_true",
    help="Skip alignment; use raw coordinates.",
)
parser.add_argument(
    "--color-by-ss",
    action="store_true",
    help="Color by secondary structure instead of per-object colors.",
)
parser.add_argument(
    "--opacity",
    type=float,
    default=1.0,
    metavar="FLOAT",
    help="Cartoon opacity 0.0–1.0 (default: 1.0).",
)

args = parser.parse_args(argv)
main(
    output_gif=args.output_gif,
    structure_files=args.structure_files,
    hold_first=args.hold_first,
    hold_last=args.hold_last,
    duration=args.duration,
    width=args.width,
    height=args.height,
    dpi=args.dpi,
    loop=args.loop,
    align=not args.no_align,
    color_by_ss=args.color_by_ss,
    opacity=args.opacity,
)
