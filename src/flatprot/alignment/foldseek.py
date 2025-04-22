# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, NamedTuple
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

from flatprot.transformation import TransformationMatrix


class AlignmentResult(NamedTuple):
    """Results from a structural family alignment."""

    db_id: str
    probability: float
    aligned_region: np.ndarray
    alignment_scores: np.ndarray
    rotation_matrix: TransformationMatrix


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
            tmp_dir = tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True, prefix="foldseek_"
            )
            tmp_dir_path = Path(tmp_dir.name)
        else:
            tmp_dir_path = tmp_dir

        # Run FoldSeek search
        result_file = tmp_dir_path / "foldseek_result.tsv"
        foldseek_tmp_dir = tmp_dir_path / "foldseek_tmp"

        self._run_foldseek_search(structure_path, result_file, foldseek_tmp_dir)

        if not result_file.exists():
            raise RuntimeError("FoldSeek result file not found")

        # Parse results
        results = pl.read_csv(
            result_file,
            separator="\t",
            has_header=False,
            new_columns=[
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
            schema_overrides={
                "query": pl.Utf8,
                "target": pl.Utf8,
            },
        )
        if len(results) == 0:
            return None

        # Get best match (or fixed family if specified)
        if fixed_alignment_id:
            match = results.filter(pl.col("target") == fixed_alignment_id)
        else:
            match = results.sort("prob", descending=True)
            if match[0, "prob"] < min_probability:
                return None

        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup()

        target_to_query_matrix = _parse_foldseek_vector(match[0, "u"]).reshape(3, 3)
        target_to_query_translation = _parse_foldseek_vector(match[0, "t"])

        # Manually calculate the inverse of the alignment transformation
        # Inverse rotation is the transpose of the rotation matrix
        R_align_inv = target_to_query_matrix.T
        # Inverse translation is -R_inv @ t
        t_align_inv = -R_align_inv @ target_to_query_translation
        query_to_target_transform = TransformationMatrix(
            rotation=R_align_inv, translation=t_align_inv
        )

        return AlignmentResult(
            db_id=match[0, "target"],
            probability=float(match[0, "prob"]),
            aligned_region=np.array((int(match[0, "qstart"]), int(match[0, "qend"]))),
            alignment_scores=_parse_foldseek_vector(match[0, "lddtfull"]),
            rotation_matrix=query_to_target_transform,
        )

    def _run_foldseek_search(
        self, structure_path: Path, output_file: Path, tmp_dir: Path
    ) -> None:
        """Runs FoldSeek search against family database."""
        cmd = [
            self.foldseek_executable,
            "easy-search",
            str(structure_path),
            str(self.database_path),
            str(output_file),
            str(tmp_dir),
            "--format-output",
            "query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,u,t,lddtfull",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_foldseek_vector(vector_str: str) -> np.ndarray:
    """Parse vector string in format '(x, y, z)' or '(x, y, z, w, ...)' to numpy array."""
    # Remove parentheses and split by comma
    nums = vector_str.strip("()")
    # Convert to float array
    return np.fromstring(nums, sep=",")
