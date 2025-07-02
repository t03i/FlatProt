#!/usr/bin/env python3

import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
import re


def extract_sequence_from_cif(cif_path: str) -> str:
    """Extract protein sequence from CIF file."""
    try:
        with open(cif_path, "r") as f:
            content = f.read()

        # Look for the sequence line
        sequence_match = re.search(
            r"_entity_poly\.pdbx_seq_one_letter_code\s+([A-Z\s]+)", content
        )
        if sequence_match:
            # Remove whitespace and return sequence
            sequence = re.sub(r"\s+", "", sequence_match.group(1))
            return sequence

        # Fallback: try to extract from ATOM records
        lines = content.split("\n")
        residues = {}
        for line in lines:
            if line.startswith("ATOM"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        res_num = int(parts[8])
                        res_name = parts[5]
                        # Convert three-letter to one-letter codes
                        aa_map = {
                            "ALA": "A",
                            "ARG": "R",
                            "ASN": "N",
                            "ASP": "D",
                            "CYS": "C",
                            "GLN": "Q",
                            "GLU": "E",
                            "GLY": "G",
                            "HIS": "H",
                            "ILE": "I",
                            "LEU": "L",
                            "LYS": "K",
                            "MET": "M",
                            "PHE": "F",
                            "PRO": "P",
                            "SER": "S",
                            "THR": "T",
                            "TRP": "W",
                            "TYR": "Y",
                            "VAL": "V",
                        }
                        if res_name in aa_map:
                            residues[res_num] = aa_map[res_name]
                    except (ValueError, IndexError):
                        continue

        if residues:
            # Sort by residue number and join
            sorted_residues = sorted(residues.items())
            return "".join([aa for _, aa in sorted_residues])

    except Exception as e:
        print(f"Warning: Could not extract sequence from {cif_path}: {e}")

    return ""


def create_hydrophobic_visualization(
    structure_path: str,
    output_path: str,
    canvas_width: int = 800,
    canvas_height: int = 600,
    show_positions: str = "minimal",
) -> None:
    """Create a complete hydrophobic gradient visualization."""

    structure_path = Path(structure_path)
    output_path = Path(output_path)

    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate base SVG with flatprot
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
        tmp_svg_path = tmp_svg.name

    try:
        # Run flatprot project
        cmd = [
            "uv",
            "run",
            "flatprot",
            "project",
            str(structure_path),
            tmp_svg_path,
            "--canvas-width",
            str(canvas_width),
            "--canvas-height",
            str(canvas_height),
            "--show-positions",
            show_positions,
            "--quiet",
        ]

        print(f"🚀 Generating base SVG from {structure_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FlatProt failed: {result.stderr}")

        # Extract sequence
        sequence = extract_sequence_from_cif(structure_path)
        if not sequence:
            print("Warning: Could not extract sequence, using default values")
            sequence = "A" * 100  # Fallback

        # Read the SVG content
        with open(tmp_svg_path, "r") as f:
            svg_content = f.read()

        # Create the complete visualization with hydrophobic gradients
        html_content = create_hydrophobic_html(
            svg_content, sequence, structure_path.stem, canvas_width, canvas_height
        )

        # Write the final HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"✅ Hydrophobic gradient visualization created: {output_path}")
        print(f"🧬 Sequence length: {len(sequence)} residues")

    finally:
        # Clean up temporary file
        Path(tmp_svg_path).unlink(missing_ok=True)


def create_hydrophobic_html(
    svg_content: str,
    sequence: str,
    structure_name: str,
    canvas_width: int,
    canvas_height: int,
) -> str:
    """Create complete HTML with hydrophobic gradients and legend."""

    # Hydrophobicity scale (Kyte-Doolittle normalized to 0-1)
    hydrophobicity_raw = {
        "A": 1.8,
        "R": -4.5,
        "N": -3.5,
        "D": -3.5,
        "C": 2.5,
        "Q": -3.5,
        "E": -3.5,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "L": 3.8,
        "K": -3.9,
        "M": 1.9,
        "F": 2.8,
        "P": -1.6,
        "S": -0.8,
        "T": -0.7,
        "W": -0.9,
        "Y": -1.3,
        "V": 4.2,
    }

    # Normalize to 0-1 range
    min_val = min(hydrophobicity_raw.values())
    max_val = max(hydrophobicity_raw.values())
    hydrophobicity = {
        aa: (val - min_val) / (max_val - min_val)
        for aa, val in hydrophobicity_raw.items()
    }

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hydrophobic Gradient: {structure_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .container {{
            max-width: {canvas_width + 400}px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
            font-weight: 300;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 16px;
        }}
        .content {{
            display: flex;
            flex-wrap: wrap;
            align-items: flex-start;
        }}
        .svg-panel {{
            flex: 1;
            min-width: {canvas_width}px;
            padding: 30px;
            text-align: center;
        }}
        .legend-panel {{
            width: 300px;
            padding: 30px;
            background: #f8f9fa;
            border-left: 1px solid #e9ecef;
        }}
        .svg-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .legend {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            margin: 0 0 15px 0;
            color: #495057;
            font-size: 18px;
        }}
        .color-scale {{
            height: 40px;
            border-radius: 6px;
            background: linear-gradient(to right,
                rgb(0,0,255) 0%,
                rgb(64,64,223) 20%,
                rgb(128,128,191) 40%,
                rgb(191,128,128) 60%,
                rgb(223,64,64) 80%,
                rgb(255,0,0) 100%);
            margin-bottom: 10px;
            border: 1px solid #dee2e6;
        }}
        .scale-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 20px;
        }}
        .aa-examples {{
            font-size: 14px;
            line-height: 1.6;
        }}
        .aa-examples div {{
            margin: 8px 0;
            display: flex;
            align-items: center;
        }}
        .aa-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
            border: 1px solid #dee2e6;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            margin: 0 5px;
            transition: transform 0.2s;
        }}
        .btn:hover {{
            transform: translateY(-2px);
        }}
        .btn-secondary {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        }}
        .info {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 0 6px 6px 0;
            margin-top: 20px;
            font-size: 14px;
            line-height: 1.5;
        }}
        @media (max-width: 768px) {{
            .content {{ flex-direction: column; }}
            .legend-panel {{ width: 100%; border-left: none; border-top: 1px solid #e9ecef; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Hydrophobic Gradient Visualization</h1>
            <p>Structure: {structure_name} | Sequence length: {len(sequence)} residues</p>
        </div>

        <div class="content">
            <div class="svg-panel">
                <div class="svg-container" id="svg-container">
                    {svg_content}
                </div>

                <div class="controls">
                    <button class="btn" onclick="applyHydrophobicGradients()">
                        🎨 Apply Hydrophobic Gradients
                    </button>
                    <button class="btn" onclick="applyRainbowGradients()" style="background: linear-gradient(45deg, #ff4444, #ffaa00, #00ff00, #0088ff, #8800ff);">
                        🌈 Rainbow Test
                    </button>
                    <button class="btn btn-secondary" onclick="clearGradients()">
                        🗑️ Clear Gradients
                    </button>
                </div>

                <div class="info" id="info">
                    <strong>Status:</strong> Ready to apply hydrophobic gradients
                </div>
            </div>

            <div class="legend-panel">
                <div class="legend">
                    <h3>💧 Hydrophobicity Scale</h3>

                    <div class="color-scale"></div>
                    <div class="scale-labels">
                        <span>Hydrophilic</span>
                        <span>Hydrophobic</span>
                    </div>

                    <div class="aa-examples">
                        <div>
                            <div class="aa-color" style="background: rgb(0,0,255);"></div>
                            <span><strong>Hydrophilic:</strong> R, K, D, E, N, Q</span>
                        </div>
                        <div>
                            <div class="aa-color" style="background: rgb(128,128,191);"></div>
                            <span><strong>Neutral:</strong> G, S, T, H, P, Y</span>
                        </div>
                        <div>
                            <div class="aa-color" style="background: rgb(255,0,0);"></div>
                            <span><strong>Hydrophobic:</strong> I, L, V, F, W, M, A, C</span>
                        </div>
                    </div>
                </div>

                <div class="legend" style="margin-top: 20px;">
                    <h3>📊 Visualization Details</h3>
                    <div style="font-size: 14px; line-height: 1.6;">
                        <p><strong>Method:</strong> SVG mask-based gradients</p>
                        <p><strong>Scale:</strong> Kyte-Doolittle hydrophobicity</p>
                        <p><strong>Resolution:</strong> Per-residue color stops</p>
                        <p><strong>Range:</strong> Blue (hydrophilic) to Red (hydrophobic)</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const SEQUENCE = "{sequence}";

        // Normalized hydrophobicity values (0-1 range)
        const HYDROPHOBICITY = {dict(hydrophobicity)};

        function findSecondaryStructures() {{
            const svg = document.querySelector('#svg-container svg');
            const elements = [];

            if (!svg) {{
                console.error('No SVG found!');
                return elements;
            }}

            const paths = svg.querySelectorAll('path');

            paths.forEach(path => {{
                const id = path.id;
                const className = path.getAttribute('class') || '';

                if (className.includes('element')) {{
                    const match = id.match(/(Helix|Sheet)SceneElement-[A-Z]-(\\d+)-(\\d+)/);
                    if (match) {{
                        try {{
                            const bbox = path.getBBox();
                            const element = {{
                                id: id,
                                path: path,
                                type: match[1].toLowerCase(),
                                startRes: parseInt(match[2]),
                                endRes: parseInt(match[3]),
                                bbox: bbox
                            }};
                            elements.push(element);
                        }} catch (e) {{
                            console.error('Error getting bbox for', id, e);
                        }}
                    }}
                }}
            }});

            return elements;
        }}

        function createHydrophobicGradient(element) {{
            const svg = document.querySelector('#svg-container svg');
            let defs = svg.querySelector('defs');
            if (!defs) {{
                defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                svg.insertBefore(defs, svg.firstChild);
            }}

            const gradientId = `hydro-${{element.id}}`;
            const clipId = `clip-${{element.id}}`;

            // Remove existing elements
            const existingGrad = defs.querySelector(`#${{gradientId}}`);
            if (existingGrad) existingGrad.remove();
            const existingClip = defs.querySelector(`#${{clipId}}`);
            if (existingClip) existingClip.remove();

            // Create clipping path instead of mask for full opacity
            const clipPath = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
            clipPath.id = clipId;
            const clipPathElement = element.path.cloneNode(true);
            clipPathElement.removeAttribute('fill');
            clipPathElement.removeAttribute('stroke');
            clipPathElement.removeAttribute('stroke-width');
            clipPath.appendChild(clipPathElement);
            defs.appendChild(clipPath);

            // Create hydrophobic gradient
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.id = gradientId;

            const bbox = element.bbox;
            if (bbox.width > bbox.height) {{
                gradient.setAttribute('x1', '0%');
                gradient.setAttribute('y1', '50%');
                gradient.setAttribute('x2', '100%');
                gradient.setAttribute('y2', '50%');
            }} else {{
                gradient.setAttribute('x1', '50%');
                gradient.setAttribute('y1', '0%');
                gradient.setAttribute('x2', '50%');
                gradient.setAttribute('y2', '100%');
            }}

            // Create stops based on actual amino acid properties
            const residueCount = element.endRes - element.startRes + 1;
            for (let i = 0; i < residueCount; i++) {{
                const resNum = element.startRes + i;
                const seqIndex = resNum - 1;
                const aa = SEQUENCE[seqIndex] || 'A';
                const hydro = HYDROPHOBICITY[aa] || 0.3;

                // Convert hydrophobic value to color (blue = hydrophilic, red = hydrophobic)
                const r = Math.round(hydro * 255);
                const g = 0;
                const b = Math.round((1 - hydro) * 255);
                const color = `rgb(${{r}}, ${{g}}, ${{b}})`;

                const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                const offset = residueCount === 1 ? 0 : (i / (residueCount - 1)) * 100;
                stop.setAttribute('offset', `${{offset}}%`);
                stop.setAttribute('stop-color', color);
                gradient.appendChild(stop);
            }}

            defs.appendChild(gradient);

            // Create and apply gradient rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.id = `grad-rect-${{element.id}}`;
            rect.setAttribute('x', bbox.x - 10);
            rect.setAttribute('y', bbox.y - 10);
            rect.setAttribute('width', bbox.width + 20);
            rect.setAttribute('height', bbox.height + 20);
            rect.setAttribute('fill', `url(#${{gradientId}})`);
            rect.setAttribute('fill-opacity', '1.0');
            rect.setAttribute('stroke', '#000');
            rect.setAttribute('stroke-width', '1');
            rect.setAttribute('stroke-opacity', '1.0');
            rect.setAttribute('clip-path', `url(#${{clipId}})`);
            rect.classList.add('gradient-rect');

            element.path.parentNode.insertBefore(rect, element.path);

            // Hide original path completely for full opacity gradient
            element.path.style.display = 'none';

            return residueCount;
        }}

        function createRainbowGradient(element) {{
            const svg = document.querySelector('#svg-container svg');
            let defs = svg.querySelector('defs');
            if (!defs) {{
                defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                svg.insertBefore(defs, svg.firstChild);
            }}

            const gradientId = `rainbow-${{element.id}}`;
            const clipId = `clip-${{element.id}}`;

            // Remove existing elements
            const existingGrad = defs.querySelector(`#${{gradientId}}`);
            if (existingGrad) existingGrad.remove();
            const existingClip = defs.querySelector(`#${{clipId}}`);
            if (existingClip) existingClip.remove();

            // Create clipping path instead of mask for full opacity
            const clipPath = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
            clipPath.id = clipId;
            const clipPathElement = element.path.cloneNode(true);
            clipPathElement.removeAttribute('fill');
            clipPathElement.removeAttribute('stroke');
            clipPathElement.removeAttribute('stroke-width');
            clipPath.appendChild(clipPathElement);
            defs.appendChild(clipPath);

            // Create rainbow gradient
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.id = gradientId;

            const bbox = element.bbox;
            if (bbox.width > bbox.height) {{
                gradient.setAttribute('x1', '0%');
                gradient.setAttribute('y1', '50%');
                gradient.setAttribute('x2', '100%');
                gradient.setAttribute('y2', '50%');
            }} else {{
                gradient.setAttribute('x1', '50%');
                gradient.setAttribute('y1', '0%');
                gradient.setAttribute('x2', '50%');
                gradient.setAttribute('y2', '100%');
            }}

            // Create rainbow stops for each residue
            const residueCount = element.endRes - element.startRes + 1;
            const rainbowColors = [
                '#ff0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00', '#ffff00',
                '#ccff00', '#99ff00', '#66ff00', '#33ff00', '#00ff00', '#00ff33',
                '#00ff66', '#00ff99', '#00ffcc', '#00ffff', '#00ccff', '#0099ff',
                '#0066ff', '#0033ff', '#0000ff', '#3300ff', '#6600ff', '#9900ff'
            ];

            for (let i = 0; i < residueCount; i++) {{
                const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                const offset = residueCount === 1 ? 0 : (i / (residueCount - 1)) * 100;
                const colorIndex = Math.floor((i / residueCount) * rainbowColors.length);
                const color = rainbowColors[Math.min(colorIndex, rainbowColors.length - 1)];

                stop.setAttribute('offset', `${{offset}}%`);
                stop.setAttribute('stop-color', color);
                gradient.appendChild(stop);
            }}

            defs.appendChild(gradient);

            // Create and apply gradient rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.id = `grad-rect-${{element.id}}`;
            rect.setAttribute('x', bbox.x - 10);
            rect.setAttribute('y', bbox.y - 10);
            rect.setAttribute('width', bbox.width + 20);
            rect.setAttribute('height', bbox.height + 20);
            rect.setAttribute('fill', `url(#${{gradientId}})`);
            rect.setAttribute('fill-opacity', '1.0');
            rect.setAttribute('stroke', '#000');
            rect.setAttribute('stroke-width', '1');
            rect.setAttribute('stroke-opacity', '1.0');
            rect.setAttribute('clip-path', `url(#${{clipId}})`);
            rect.classList.add('gradient-rect');

            element.path.parentNode.insertBefore(rect, element.path);

            // Hide original path completely for full opacity gradient
            element.path.style.display = 'none';

            return residueCount;
        }}

        function applyHydrophobicGradients() {{
            clearGradients();
            const elements = findSecondaryStructures();

            if (elements.length === 0) {{
                updateInfo('❌ No secondary structure elements detected!');
                return;
            }}

            let totalResidues = 0;
            elements.forEach(element => {{
                totalResidues += createHydrophobicGradient(element);
            }});

            updateInfo(`✅ Hydrophobic gradients applied! ${{totalResidues}} residues across ${{elements.length}} structures with full opacity`);
        }}

        function applyRainbowGradients() {{
            clearGradients();
            const elements = findSecondaryStructures();

            if (elements.length === 0) {{
                updateInfo('❌ No secondary structure elements detected!');
                return;
            }}

            let totalResidues = 0;
            elements.forEach(element => {{
                totalResidues += createRainbowGradient(element);
            }});

            updateInfo(`🌈 Rainbow gradients applied! ${{totalResidues}} residues across ${{elements.length}} structures - perfect for testing gradient stops`);
        }}

        function clearGradients() {{
            // Remove all gradient rectangles
            document.querySelectorAll('.gradient-rect').forEach(rect => rect.remove());

            // Restore original paths
            const elements = findSecondaryStructures();
            elements.forEach(element => {{
                element.path.style.display = '';
                element.path.style.fillOpacity = '1';
                element.path.style.strokeOpacity = '1';
                if (element.type === 'helix') {{
                    element.path.style.fill = '#f00';
                    element.path.style.stroke = '#000';
                    element.path.style.strokeWidth = '1';
                }} else {{
                    element.path.style.fill = '#00f';
                    element.path.style.stroke = '#000';
                    element.path.style.strokeWidth = '1';
                }}
            }});

            updateInfo('🗑️ Gradients cleared - original FlatProt colors restored');
        }}

        function updateInfo(message) {{
            document.getElementById('info').innerHTML = `<strong>Status:</strong> ${{message}}`;
        }}

        // Auto-apply gradients on load
        setTimeout(() => {{
            applyHydrophobicGradients();
        }}, 1000);
    </script>
</body>
</html>"""

    return html_template


def main():
    parser = argparse.ArgumentParser(
        description="Create hydrophobic gradient visualization from protein structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hydrophobic_gradient_viz.py structure.cif output.html
  python hydrophobic_gradient_viz.py structure.pdb output.html --width 1000 --height 800
  python hydrophobic_gradient_viz.py structure.cif output.html --positions none
        """,
    )

    parser.add_argument("structure", help="Input structure file (CIF, PDB, etc.)")
    parser.add_argument("output", help="Output HTML file path")
    parser.add_argument(
        "--width", type=int, default=800, help="Canvas width in pixels (default: 800)"
    )
    parser.add_argument(
        "--height", type=int, default=600, help="Canvas height in pixels (default: 600)"
    )
    parser.add_argument(
        "--positions",
        choices=["major", "minor", "none"],
        default="minimal",
        help="Position annotation level (default: major)",
    )

    args = parser.parse_args()

    try:
        create_hydrophobic_visualization(
            structure_path=args.structure,
            output_path=args.output,
            canvas_width=args.width,
            canvas_height=args.height,
            show_positions=args.positions,
        )

        print("\n🎉 Success! Open the visualization:")
        print(f"   {Path(args.output).absolute()}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
