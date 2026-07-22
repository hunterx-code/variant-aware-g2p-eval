from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE_ROOT))

from run_evaluation import (  # noqa: E402
    build_report_database,
    levenshtein,
    query_report_datasets,
    score_prediction,
    summarize_group,
)


class EvaluationTests(unittest.TestCase):
    def test_levenshtein_counts_phone_edits(self) -> None:
        self.assertEqual(levenshtein(["K", "AE1", "T"], ["K", "AE1", "T"]), 0)
        self.assertEqual(levenshtein(["K", "AE1", "T"], ["K", "AH0", "T"]), 1)
        self.assertEqual(levenshtein(["K", "AE1"], ["K", "AE1", "T"]), 1)

    def test_variant_aware_scoring_rescues_nonfirst_exact_match(self) -> None:
        prediction = ["AE1", "L", "IH0", "S"]
        references = [
            ["AE1", "L", "AH0", "S"],
            ["AE1", "L", "IH0", "S"],
        ]
        result = score_prediction(prediction, references)
        self.assertFalse(result["strict_exact"])
        self.assertTrue(result["variant_exact"])
        self.assertTrue(result["rescued_by_variant_scoring"])
        self.assertEqual(result["min_edit_distance"], 0)
        self.assertEqual(result["min_per"], 0)

    def test_edit_distance_and_per_are_minimized_independently(self) -> None:
        prediction = ["A", "B", "C", "D", "E", "F"]
        references = [
            ["A", "B", "C", "D", "E", "X", "G", "H", "I"],
            ["A", "B", "C", "X", "Y", "Z"],
        ]
        result = score_prediction(prediction, references)
        self.assertEqual(result["min_edit_distance"], 3)
        self.assertAlmostEqual(result["min_per"], 4 / 9)
        self.assertEqual(result["closest_reference"], references[0])

    def test_group_summary_uses_name_count_as_denominator(self) -> None:
        rows = [
            {
                "strict_exact": True,
                "variant_exact": True,
                "within_one_phone": True,
                "per_at_most_25pct": True,
                "min_per": 0.0,
                "rescued_by_variant_scoring": False,
            },
            {
                "strict_exact": False,
                "variant_exact": True,
                "within_one_phone": True,
                "per_at_most_25pct": True,
                "min_per": 0.2,
                "rescued_by_variant_scoring": True,
            },
        ]
        summary = summarize_group(rows, "test")
        self.assertEqual(summary["sample_size"], 2)
        self.assertEqual(summary["strict_exact"], 0.5)
        self.assertEqual(summary["variant_exact"], 1.0)
        self.assertEqual(summary["variant_rescues"], 1)
        self.assertAlmostEqual(summary["mean_min_per"], 0.1)

    def test_empty_reference_set_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            score_prediction(["K"], [])

    def test_report_datasets_are_produced_by_auditable_sql(self) -> None:
        rows = [
            {
                "name": "alice",
                "prediction": ["AE1", "L", "IH0", "S"],
                "references": [["AE1", "L", "AH0", "S"], ["AE1", "L", "IH0", "S"]],
                "reference_count": 2,
                "strict_exact": False,
                "variant_exact": True,
                "rescued_by_variant_scoring": True,
                "min_edit_distance": 0,
                "min_per": 0.0,
                "within_one_phone": True,
                "per_at_most_25pct": True,
                "closest_reference": ["AE1", "L", "IH0", "S"],
            },
            {
                "name": "bob",
                "prediction": ["B", "AA1", "B"],
                "references": [["B", "AA1", "B"]],
                "reference_count": 1,
                "strict_exact": True,
                "variant_exact": True,
                "rescued_by_variant_scoring": False,
                "min_edit_distance": 0,
                "min_per": 0.0,
                "within_one_phone": True,
                "per_at_most_25pct": True,
                "closest_reference": ["B", "AA1", "B"],
            },
        ]
        quality_rows = [
            {
                "check": "Fixture integrity",
                "status": "PASS",
                "observed": "2 rows",
                "decision_risk": "Test-only fixture",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation.sqlite"
            build_report_database(path, rows, quality_rows)
            datasets = query_report_datasets(path)

        self.assertEqual(datasets["headline"][0]["evaluated_names"], 2)
        self.assertEqual(datasets["headline"][0]["variant_rescues"], 1)
        self.assertEqual(len(datasets["scoring_by_cohort"]), 8)
        self.assertEqual(datasets["variant_rescues"][0]["name"], "Alice")
        self.assertEqual(datasets["data_quality"][0]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
