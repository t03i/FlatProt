# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

from flatprot.transformation import TransformationMatrix
from .utils import AlignmentResult


class FoldseekAligner:
    """Handles structural family alignments using FoldSeek."""

    def __init__(
        self,
        foldseek_executable: str,
        database_path: Path,
    ):
        self.foldseek_executable = foldseek_executable
        self.database_path = database_path

    def align_structure(
        self,
        structure_path: Path,
        min_probability: float = 0.5,
        fixed_alignment_id: Optional[str] = None,
        tmp_dir: Optional[Path] = None,
    ) -> Optional[AlignmentResult]:
        """Aligns structure to family database and returns best match."""

        if tmp_dir is None:
            tmp_dir = tempfile.TemporaryDirectory()
            tmp_dir_path = Path(tmp_dir.name)
        else:
            tmp_dir_path = tmp_dir

        # Run FoldSeek search
        result_file = tmp_dir_path / "foldseek_result.tsv"
        foldseek_tmp_dir = tmp_dir_path / "foldseek_tmp"

        self._run_foldseek_search(structure_path, result_file, foldseek_tmp_dir)

        # Parse results
        results = pl.read_csv(
            result_file,
            sep="\t",
            names=[
                "query",
                "target",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "tseq",
                "prob",
                "alntmscore",
                "u",
                "t",
                "lddtfull",
            ],
        )

        if len(results) == 0:
            return None

        # Get best match (or fixed family if specified)
        if fixed_alignment_id:
            match = results[results["target"] == fixed_alignment_id].iloc[0]
        else:
            match = results.sort_values("prob", descending=True).iloc[0]
            if match["prob"] < min_probability:
                return None

        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup()

        rotation_matrix = TransformationMatrix(
            rotation=_parse_foldseek_vector(match["u"]).reshape(3, 3),
            translation=_parse_foldseek_vector(match["t"]),
        )

        return AlignmentResult(
            family_id=match["target"],
            probability=float(match["prob"]),
            aligned_region=np.array((int(match["qstart"]), int(match["qend"]))),
            lddt_scores=_parse_foldseek_vector(match["lddtfull"]),
            rotation_matrix=rotation_matrix,
        )

    def _run_foldseek_search(self, structure_path: Path, output_file: Path) -> None:
        """Runs FoldSeek search against family database."""
        cmd = [
            self.foldseek_executable,
            "easy-search",
            str(structure_path),
            str(self.database_path),
            str(output_file),
            str(self.tmp_dir),
            "--format-output",
            "query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,u,t,lddtfull",
        ]
        subprocess.run(cmd, check=True)


def _parse_foldseek_vector(vector_str: str) -> np.ndarray:
    """Parse vector string in format '(x, y, z)' or '(x, y, z, w, ...)' to numpy array."""
    # Remove parentheses and split by comma
    nums = vector_str.strip("()").split(",")
    # Convert to float array
    return np.fromstring(nums, sep=",")
