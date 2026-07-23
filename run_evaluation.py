#!/usr/bin/env python3
"""Build a contamination-aware, variant-aware G2P evaluation snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


NLTK_DATA_COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
SOURCE_SPECS = {
    "cmudict": {
        "url": f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_COMMIT}/packages/corpora/cmudict.zip",
        "sha256": "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6",
        "target": "corpora/cmudict.zip",
    },
    "names": {
        "url": f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_COMMIT}/packages/corpora/names.zip",
        "sha256": "0eec7e958b34982662b8f05824ae64642dea097b08057ade65c252191c5fe7ca",
        "target": "corpora/names.zip",
    },
    "averaged_perceptron_tagger": {
        "url": f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_COMMIT}/packages/taggers/averaged_perceptron_tagger.zip",
        "sha256": "e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9",
        "target": "taggers/averaged_perceptron_tagger.zip",
    },
}


HEADLINE_SQL = """\
WITH overall AS (
    SELECT
        COUNT(*) AS evaluated_names,
        AVG(strict_exact) AS strict_exact,
        AVG(variant_exact) AS variant_exact,
        SUM(rescued_by_variant_scoring) AS variant_rescues
    FROM evaluation_rows
),
multi_reference AS (
    SELECT
        AVG(strict_exact) AS multi_strict_exact,
        AVG(variant_exact) AS multi_variant_exact
    FROM evaluation_rows
    WHERE reference_count > 1
)
SELECT
    overall.evaluated_names,
    overall.strict_exact,
    overall.variant_exact,
    multi_reference.multi_strict_exact,
    multi_reference.multi_variant_exact,
    multi_reference.multi_variant_exact - multi_reference.multi_strict_exact AS multi_delta,
    overall.variant_rescues
FROM overall
CROSS JOIN multi_reference
"""


COHORT_CHART_SQL = """\
WITH cohort_summary AS (
    SELECT
        CASE
            WHEN reference_count = 1 THEN 'Single reference'
            ELSE 'Multiple references'
        END AS cohort,
        COUNT(*) AS sample_size,
        SUM(strict_exact) AS strict_count,
        SUM(variant_exact) AS variant_count,
        SUM(within_one_phone) AS one_phone_count,
        SUM(per_at_most_25pct) AS per_25_count
    FROM evaluation_rows
    GROUP BY CASE
        WHEN reference_count = 1 THEN 'Single reference'
        ELSE 'Multiple references'
    END
),
metric_rows AS (
    SELECT cohort, 'Canonical exact' AS metric,
           1.0 * strict_count / sample_size AS rate,
           sample_size, strict_count AS numerator, 1 AS metric_order
    FROM cohort_summary
    UNION ALL
    SELECT cohort, 'Any-reference exact',
           1.0 * variant_count / sample_size,
           sample_size, variant_count, 2
    FROM cohort_summary
    UNION ALL
    SELECT cohort, 'Within one phone',
           1.0 * one_phone_count / sample_size,
           sample_size, one_phone_count, 3
    FROM cohort_summary
    UNION ALL
    SELECT cohort, 'PER at most 25%',
           1.0 * per_25_count / sample_size,
           sample_size, per_25_count, 4
    FROM cohort_summary
)
SELECT cohort, metric, rate, sample_size, numerator
FROM metric_rows
ORDER BY metric_order,
         CASE cohort WHEN 'Single reference' THEN 1 ELSE 2 END
"""


VARIANT_RESCUES_SQL = """\
SELECT
    display_name AS name,
    prediction,
    canonical_reference,
    closest_reference AS matched_variant,
    reference_count
FROM evaluation_rows
WHERE rescued_by_variant_scoring = 1
ORDER BY name
LIMIT 12
"""


WORST_CASES_SQL = """\
SELECT
    display_name AS name,
    prediction,
    closest_reference,
    min_per,
    reference_count
FROM evaluation_rows
ORDER BY min_per DESC, name
LIMIT 12
"""


QUALITY_CHECKS_SQL = """\
SELECT check_name AS 'check', status, observed, decision_risk
FROM quality_checks
ORDER BY check_name
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(cache_dir: Path, name: str, spec: dict[str, str]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{name}-{NLTK_DATA_COMMIT}.zip"
    if not destination.exists():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "60",
                spec["url"],
                "--output",
                str(destination),
            ],
            check=True,
        )
    actual = sha256_file(destination)
    if actual != spec["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {name}: expected {spec['sha256']}, got {actual}"
        )
    return destination


def install_nltk_archives(cache_dir: Path, nltk_root: Path) -> dict[str, Path]:
    installed: dict[str, Path] = {}
    for name, spec in SOURCE_SPECS.items():
        source = ensure_source(cache_dir, name, spec)
        target = nltk_root / spec["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        installed[name] = source
    return installed


def levenshtein(left: Iterable[str], right: Iterable[str]) -> int:
    a, b = list(left), list(right)
    previous = list(range(len(b) + 1))
    for index, left_value in enumerate(a, 1):
        current = [index]
        for column, right_value in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def score_prediction(
    prediction: list[str], references: list[list[str]]
) -> dict[str, Any]:
    """Score references, keeping raw-distance and PER minima independent.

    ``closest_reference`` follows minimum PER, then raw distance and source
    order for deterministic ties.
    """
    if not references or any(not reference for reference in references):
        raise ValueError("Every row needs at least one non-empty reference")
    distances = [levenshtein(prediction, reference) for reference in references]
    pers = [distance / len(reference) for distance, reference in zip(distances, references)]
    best_per_index = min(
        range(len(references)), key=lambda idx: (pers[idx], distances[idx], idx)
    )
    min_edit_distance = min(distances)
    strict_exact = prediction == references[0]
    variant_exact = prediction in references
    return {
        "strict_exact": strict_exact,
        "variant_exact": variant_exact,
        "rescued_by_variant_scoring": variant_exact and not strict_exact,
        "min_edit_distance": min_edit_distance,
        "min_per": pers[best_per_index],
        "within_one_phone": min_edit_distance <= 1,
        "per_at_most_25pct": pers[best_per_index] <= 0.25,
        "closest_reference": references[best_per_index],
    }


def summarize_group(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    count = len(rows)
    if not count:
        raise ValueError(f"Cannot summarize empty cohort: {label}")
    rate_fields = (
        "strict_exact",
        "variant_exact",
        "within_one_phone",
        "per_at_most_25pct",
    )
    return {
        "cohort": label,
        "sample_size": count,
        **{field: sum(bool(row[field]) for row in rows) / count for field in rate_fields},
        "mean_min_per": sum(float(row["min_per"]) for row in rows) / count,
        "variant_rescues": sum(bool(row["rescued_by_variant_scoring"]) for row in rows),
    }


def compact_phones(phones: Iterable[str]) -> str:
    return " ".join(phones)


def build_quality_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "check": "Normalized-name uniqueness",
            "status": "PASS",
            "observed": f"{summary['population']['duplicate_rows_removed']} duplicate rows collapsed",
            "decision_risk": "Prevents duplicate weighting across the two source lists.",
        },
        {
            "check": "Letters-only eligibility",
            "status": "CAVEAT",
            "observed": (
                f"{summary['population']['spellings_excluded_by_letters_only_filter']} "
                "spellings excluded"
            ),
            "decision_risk": (
                "This demo excludes spellings with spaces, punctuation, or "
                "characters outside a-z."
            ),
        },
        {
            "check": "Dictionary join coverage",
            "status": "CAVEAT",
            "observed": f"{summary['population']['join_coverage']:.1%} of eligible names",
            "decision_risk": "Missing names are not random; reported rates cannot represent all names.",
        },
        {
            "check": "Reference integrity",
            "status": "PASS",
            "observed": "No empty or duplicate pronunciation variants",
            "decision_risk": "Each scored row has at least one usable reference.",
        },
        {
            "check": "Model/reference independence",
            "status": "BLOCK",
            "observed": "g2p-en training code loads CMUdict",
            "decision_risk": "Do not use these rates to rank or select a model.",
        },
    ]


def build_report_database(
    database_path: Path,
    rows: list[dict[str, Any]],
    quality_rows: list[dict[str, str]],
) -> None:
    if database_path.exists():
        database_path.unlink()
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE evaluation_rows (
                name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                canonical_reference TEXT NOT NULL,
                closest_reference TEXT NOT NULL,
                reference_count INTEGER NOT NULL CHECK (reference_count >= 1),
                strict_exact INTEGER NOT NULL CHECK (strict_exact IN (0, 1)),
                variant_exact INTEGER NOT NULL CHECK (variant_exact IN (0, 1)),
                rescued_by_variant_scoring INTEGER NOT NULL CHECK (rescued_by_variant_scoring IN (0, 1)),
                min_edit_distance INTEGER NOT NULL CHECK (min_edit_distance >= 0),
                min_per REAL NOT NULL CHECK (min_per >= 0),
                within_one_phone INTEGER NOT NULL CHECK (within_one_phone IN (0, 1)),
                per_at_most_25pct INTEGER NOT NULL CHECK (per_at_most_25pct IN (0, 1))
            );
            CREATE TABLE quality_checks (
                check_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                observed TEXT NOT NULL,
                decision_risk TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO evaluation_rows (
                name, display_name, prediction, canonical_reference,
                closest_reference, reference_count, strict_exact, variant_exact,
                rescued_by_variant_scoring, min_edit_distance, min_per,
                within_one_phone, per_at_most_25pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["name"],
                    row["name"].title(),
                    compact_phones(row["prediction"]),
                    compact_phones(row["references"][0]),
                    compact_phones(row["closest_reference"]),
                    row["reference_count"],
                    int(row["strict_exact"]),
                    int(row["variant_exact"]),
                    int(row["rescued_by_variant_scoring"]),
                    row["min_edit_distance"],
                    row["min_per"],
                    int(row["within_one_phone"]),
                    int(row["per_at_most_25pct"]),
                )
                for row in rows
            ],
        )
        connection.executemany(
            """
            INSERT INTO quality_checks (check_name, status, observed, decision_risk)
            VALUES (?, ?, ?, ?)
            """,
            [
                (row["check"], row["status"], row["observed"], row["decision_risk"])
                for row in quality_rows
            ],
        )
        connection.commit()


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(sql).fetchall()]


def query_report_datasets(database_path: Path) -> dict[str, list[dict[str, Any]]]:
    with closing(sqlite3.connect(database_path)) as connection:
        return {
            "headline": query_rows(connection, HEADLINE_SQL),
            "scoring_by_cohort": query_rows(connection, COHORT_CHART_SQL),
            "variant_rescues": query_rows(connection, VARIANT_RESCUES_SQL),
            "worst_cases": query_rows(connection, WORST_CASES_SQL),
            "data_quality": query_rows(connection, QUALITY_CHECKS_SQL),
        }


def build_artifact(
    summary: dict[str, Any], datasets: dict[str, list[dict[str, Any]]], generated_at: str
) -> dict[str, Any]:
    groups = {group["cohort"]: group for group in summary["cohorts"]}
    single = groups["Single reference"]
    multiple = groups["Multiple references"]
    all_names = groups["All names"]
    multi_delta = multiple["variant_exact"] - multiple["strict_exact"]

    headline = datasets["headline"]
    chart_rows = datasets["scoring_by_cohort"]
    rescue_rows = datasets["variant_rescues"]
    worst_rows = datasets["worst_cases"]
    quality_rows = datasets["data_quality"]

    headline_source = {
        "id": "headline_query",
        "label": "Aggregate evaluation metrics",
        "path": "evaluation.sqlite",
        "query": {"sql": HEADLINE_SQL},
    }
    cohort_source = {
        "id": "cohort_query",
        "label": "Reference-cohort scoring metrics",
        "path": "evaluation.sqlite",
        "query": {"sql": COHORT_CHART_SQL},
    }
    rescue_source = {
        "id": "variant_rescue_query",
        "label": "Non-first reference exact matches",
        "path": "evaluation.sqlite",
        "query": {"sql": VARIANT_RESCUES_SQL},
    }
    worst_source = {
        "id": "worst_case_query",
        "label": "Largest minimum phoneme error rates",
        "path": "evaluation.sqlite",
        "query": {"sql": WORST_CASES_SQL},
    }
    quality_source = {
        "id": "quality_query",
        "label": "Data and claim gates",
        "path": "evaluation.sqlite",
        "query": {"sql": QUALITY_CHECKS_SQL},
    }
    sources = [
        headline_source,
        cohort_source,
        rescue_source,
        worst_source,
        quality_source,
        {
            "id": "evaluation_code",
            "label": "Reproducible evaluator and report builder",
            "path": "run_evaluation.py",
        },
        {
            "id": "nltk_names",
            "label": "NLTK names corpus snapshot",
            "href": SOURCE_SPECS["names"]["url"],
        },
        {
            "id": "nltk_cmudict",
            "label": "NLTK CMUdict snapshot",
            "href": SOURCE_SPECS["cmudict"]["url"],
        },
        {
            "id": "g2p_en_training",
            "label": "g2p-en training source showing CMUdict input",
            "href": "https://github.com/Kyubyong/g2p/blob/87bea58193f1ed451a8edc77fc6848564a243820/g2p_en/train.py",
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Variant-Aware G2P Evaluation",
            "description": "A contamination-aware pronunciation benchmark demonstration.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "names_card",
                    "description": "Normalized first-name spellings with at least one CMUdict reference.",
                    "dataset": "headline",
                    "sourceId": "headline_query",
                    "metrics": [
                        {"label": "Names evaluated", "field": "evaluated_names", "format": "number"}
                    ],
                },
                {
                    "id": "overall_card",
                    "description": "Exact-match rate when every listed pronunciation is accepted.",
                    "dataset": "headline",
                    "sourceId": "headline_query",
                    "metrics": [
                        {"label": "Any-reference exact", "field": "variant_exact", "format": "percent"},
                        {"label": "First-reference exact", "field": "strict_exact", "format": "percent"},
                    ],
                },
                {
                    "id": "multi_card",
                    "description": "Exact-match rate among names with multiple listed references.",
                    "dataset": "headline",
                    "sourceId": "headline_query",
                    "metrics": [
                        {"label": "Multi-reference exact", "field": "multi_variant_exact", "format": "percent"},
                        {"label": "First-reference exact", "field": "multi_strict_exact", "format": "percent"},
                        {"label": "Variant-aware lift", "field": "multi_delta", "format": "percent", "signed": True},
                    ],
                },
                {
                    "id": "rescue_card",
                    "description": "Predictions falsely rejected when only the first reference is treated as valid.",
                    "dataset": "headline",
                    "sourceId": "headline_query",
                    "metrics": [
                        {"label": "Variant rescues", "field": "variant_rescues", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "cohort_accuracy_chart",
                    "title": "Evaluation pass rates by reference cohort",
                    "subtitle": (
                        f"{single['sample_size']:,} single-reference and {multiple['sample_size']:,} "
                        f"multi-reference names; all rates use one {all_names['sample_size']:,}-name snapshot."
                    ),
                    "intent": "comparison",
                    "question": "How much does accepting every listed pronunciation change measured G2P performance?",
                    "rationale": "Grouped bars keep the two cohorts and four same-unit pass criteria directly comparable.",
                    "comparisonContext": {
                        "baseline": "First CMUdict reference",
                        "denominator": "Eligible names in each reference cohort",
                        "grain": "One normalized name spelling",
                        "unit": "Share of names",
                    },
                    "type": "bar",
                    "dataset": "scoring_by_cohort",
                    "sourceId": "cohort_query",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "Scoring criterion"},
                        "y": {"field": "rate", "type": "quantitative", "label": "Pass rate", "format": "percent"},
                        "color": {"field": "cohort", "type": "nominal", "label": "Reference cohort"},
                        "tooltip": [
                            {"field": "sample_size", "type": "quantitative", "label": "Names", "format": "number"},
                            {"field": "numerator", "type": "quantitative", "label": "Passing names", "format": "number"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                    "labels": {"values": "all"},
                    "legend": {"position": "bottom", "sort": "spec", "title": "Reference cohort"},
                    "palette": {"kind": "categorical"},
                    "settings": {"groupMode": "grouped", "sort": "none", "categoryLabelPolicy": "wrap"},
                    "maxRows": 8,
                    "surface": {"surface": "card", "viewMode": "both", "interactiveLegend": True},
                }
            ],
            "tables": [
                {
                    "id": "variant_rescues_table",
                    "title": "Examples rescued by variant-aware scoring",
                    "subtitle": "Alphabetical sample of predictions matching a listed non-first reference.",
                    "dataset": "variant_rescues",
                    "sourceId": "variant_rescue_query",
                    "defaultSort": {"field": "name", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "name", "label": "Name", "type": "text"},
                        {"field": "prediction", "label": "Prediction", "type": "text"},
                        {"field": "canonical_reference", "label": "First reference", "type": "text"},
                        {"field": "matched_variant", "label": "Matched variant", "type": "text"},
                        {"field": "reference_count", "label": "References", "format": "number"},
                    ],
                },
                {
                    "id": "worst_cases_table",
                    "title": "Largest phoneme disagreements",
                    "subtitle": "Twelve names with the highest minimum phoneme error rate across listed references.",
                    "dataset": "worst_cases",
                    "sourceId": "worst_case_query",
                    "defaultSort": {"field": "min_per", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "name", "label": "Name", "type": "text"},
                        {"field": "prediction", "label": "Prediction", "type": "text"},
                        {"field": "closest_reference", "label": "Closest reference", "type": "text"},
                        {"field": "min_per", "label": "Minimum PER", "format": "percent"},
                        {"field": "reference_count", "label": "References", "format": "number"},
                    ],
                },
                {
                    "id": "quality_table",
                    "title": "Data and claim gates",
                    "subtitle": "Checks that determine which conclusions this public-data demonstration can support.",
                    "dataset": "data_quality",
                    "sourceId": "quality_query",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "observed", "label": "Observed", "type": "text"},
                        {"field": "decision_risk", "label": "Why it matters", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Variant-Aware G2P Evaluation"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline_query",
                    "body": (
                        "## Technical summary\n\n"
                        "A first-reference-only scorer understated exact-match performance on ambiguous names. "
                        f"Across {summary['population']['evaluated_names']:,} normalized first-name spellings, exact match rose from "
                        f"{all_names['strict_exact']:.1%} to {all_names['variant_exact']:.1%} when every listed reference was accepted. "
                        f"For the {multiple['sample_size']} names with multiple references, the change was "
                        f"{multiple['strict_exact']:.1%} to {multiple['variant_exact']:.1%}, rescuing "
                        f"{all_names['variant_rescues']} predictions that were valid under a non-first reference.\n\n"
                        "**Decision:** the evaluator is ready as a methodology sample, but this run is deliberately blocked from model-ranking claims. "
                        "The public model's training code uses CMUdict, which is also the reference source here."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["names_card", "overall_card", "multi_card", "rescue_card"]},
                {
                    "id": "variant_finding",
                    "type": "markdown",
                    "sourceId": "cohort_query",
                    "body": (
                        "## Multiple accepted pronunciations materially change the score\n\n"
                        "Single-reference names are unchanged by construction. In the multi-reference cohort, accepting all listed variants adds "
                        f"{multi_delta:.1%} to exact-match accuracy. The softer one-phone and 25% PER gates move less because they already give near misses partial tolerance. "
                        "The implication is narrow but important: a production benchmark should separate model errors from reference-selection errors."
                    ),
                },
                {"id": "cohort_chart", "type": "chart", "chartId": "cohort_accuracy_chart"},
                {
                    "id": "rescue_explanation",
                    "type": "markdown",
                    "sourceId": "variant_rescue_query",
                    "body": (
                        "## The false rejections are inspectable, not just an aggregate\n\n"
                        "Each row below is an exact prediction that fails only because the first dictionary pronunciation is treated as canonical. "
                        "A reviewer can trace the prediction, rejected first reference, accepted variant, and reference count at the name grain."
                    ),
                },
                {"id": "rescue_table_block", "type": "table", "tableId": "variant_rescues_table"},
                {
                    "id": "failure_explanation",
                    "type": "markdown",
                    "sourceId": "worst_case_query",
                    "body": (
                        "## Large disagreements expose the model and corpus boundary\n\n"
                        "The worst rows include spellings whose pronunciation depends on language, culture, or lexicalization beyond a simple English grapheme path. "
                        "They are candidates for locale-aware references and human review, not evidence that one universal pronunciation is correct."
                    ),
                },
                {"id": "worst_table_block", "type": "table", "tableId": "worst_cases_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "headline_query",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        f"The unit is one lowercase name spelling containing only the letters a through z. The source population contains {summary['population']['raw_name_rows']:,} rows across NLTK's two first-name lists; "
                        f"normalization leaves {summary['population']['eligible_unique_names']:,} eligible unique spellings, of which {summary['population']['evaluated_names']:,} join to CMUdict ({summary['population']['join_coverage']:.1%}). "
                        "Canonical exact means equality to the first reference. Any-reference exact means equality to at least one listed reference. "
                        "PER is phoneme-level Levenshtein distance divided by reference length, minimized across references."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "evaluation_code",
                    "body": (
                        "## Methodology keeps scoring and prediction paths separate\n\n"
                        "The run unions and normalizes the two name lists, verifies source hashes, inner-joins to CMUdict, and calls `G2p.predict` directly so dictionary lookup cannot silently return the reference. "
                        "It then scores the same prediction against the first reference and against the complete reference set. Full row-level output is retained in `evaluation_rows.jsonl`; the report snapshot contains bounded aggregates and deterministic review samples."
                    ),
                },
                {"id": "quality_block", "type": "table", "tableId": "quality_table"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and robustness gates\n\n"
                        "This is an evaluator demonstration, not a population estimate or model benchmark. NLTK's name lists are dated, binary-labeled source files; those labels are intentionally discarded. CMUdict is North-American-English-oriented, does not encode the identity or locale of a specific person, and covers only part of the normalized list. "
                        "Most importantly, the public checkpoint was trained from CMUdict entries, so reference leakage is structural. The report therefore refuses a model-selection conclusion even though the calculations are reproducible."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        "Run this evaluator on a truly held-out, consented dataset with person-confirmed pronunciations, locale and language metadata, and a frozen model-training manifest. Add top-k coverage, confidence calibration, selective-risk curves, and human-review cost. Predeclare the primary metric and slices before seeing model outputs."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Which variants are interchangeable for a named person rather than merely present in a dictionary?\n"
                        "- Which locale, language, and identity slices require separate phoneme inventories?\n"
                        "- What confidence threshold minimizes harmful mispronunciation while keeping human-review volume acceptable?\n"
                        "- How should corrections persist without turning popularity into correctness?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "scoring_by_cohort": chart_rows,
                "variant_rescues": rescue_rows,
                "worst_cases": worst_rows,
                "data_quality": quality_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://variant-aware-g2p-evaluation",
            "controls": {"edit": False, "refresh": False},
        },
    }


def run(cache_dir: Path, output_dir: Path, generated_at: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="variant-aware-g2p-") as temp_dir:
        nltk_root = Path(temp_dir) / "nltk_data"
        installed = install_nltk_archives(cache_dir, nltk_root)
        os.environ["NLTK_DATA"] = str(nltk_root)

        import nltk  # Imported only after the verified local data path exists.

        nltk.data.path.insert(0, str(nltk_root))
        from g2p_en import G2p
        from nltk.corpus import cmudict, names

        raw_names = list(names.words())
        normalized_names = [value.lower() for value in raw_names]
        unique_names = set(normalized_names)
        eligible_names = {name for name in unique_names if re.fullmatch(r"[a-z]+", name)}
        reference_map = cmudict.dict()
        joined_names = sorted(eligible_names.intersection(reference_map))

        model = G2p()
        rows: list[dict[str, Any]] = []
        for name in joined_names:
            references: list[list[str]] = []
            for reference in reference_map[name]:
                if reference not in references:
                    references.append(reference)
            prediction = list(model.predict(name))
            score = score_prediction(prediction, references)
            rows.append(
                {
                    "name": name,
                    "prediction": prediction,
                    "references": references,
                    "reference_count": len(references),
                    **score,
                }
            )

    single_rows = [row for row in rows if row["reference_count"] == 1]
    multiple_rows = [row for row in rows if row["reference_count"] > 1]
    duplicate_variants = sum(
        len(value) != len({tuple(reference) for reference in value})
        for value in reference_map.values()
    )
    summary = {
        "generated_at": generated_at,
        "benchmark_status": "DEMO_ONLY_TRAINING_OVERLAP",
        "model": {
            "name": "g2p-en",
            "version": "2.1.0",
            "prediction_path": "G2p.predict",
            "claim_gate": "BLOCK_MODEL_RANKING",
            "reason": "The model training source loads CMUdict, which is also the gold-reference source.",
        },
        "population": {
            "raw_name_rows": len(raw_names),
            "unique_normalized_names": len(unique_names),
            "duplicate_rows_removed": len(raw_names) - len(unique_names),
            "spellings_excluded_by_letters_only_filter": len(unique_names) - len(eligible_names),
            "eligible_unique_names": len(eligible_names),
            "evaluated_names": len(rows),
            "join_coverage": len(rows) / len(eligible_names),
            "single_reference_names": len(single_rows),
            "multiple_reference_names": len(multiple_rows),
        },
        "quality_checks": {
            "empty_reference_entries": sum(not value for value in reference_map.values()),
            "entries_with_duplicate_variants": duplicate_variants,
            "source_hashes_verified": True,
        },
        "cohorts": [
            summarize_group(rows, "All names"),
            summarize_group(single_rows, "Single reference"),
            summarize_group(multiple_rows, "Multiple references"),
        ],
        "sources": {
            name: {
                "url": SOURCE_SPECS[name]["url"],
                "expected_sha256": SOURCE_SPECS[name]["sha256"],
                "observed_sha256": sha256_file(path),
            }
            for name, path in installed.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "evaluation_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    database_path = output_dir / "evaluation.sqlite"
    build_report_database(database_path, rows, build_quality_rows(summary))
    datasets = query_report_datasets(database_path)
    artifact = build_artifact(summary, datasets, generated_at)
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "variant-aware-g2p-eval-cache",
        help="Cache for hash-verified public source archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory for JSONL, summary, and canonical report artifact.",
    )
    parser.add_argument(
        "--generated-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        help="ISO-8601 snapshot timestamp; pass a fixed value for byte-stable output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.cache_dir, arguments.output_dir, arguments.generated_at)
    cohorts = {row["cohort"]: row for row in result["cohorts"]}
    print(
        json.dumps(
            {
                "benchmark_status": result["benchmark_status"],
                "evaluated_names": result["population"]["evaluated_names"],
                "strict_exact": cohorts["All names"]["strict_exact"],
                "variant_exact": cohorts["All names"]["variant_exact"],
                "multi_reference_variant_rescues": cohorts["Multiple references"]["variant_rescues"],
            },
            indent=2,
            sort_keys=True,
        )
    )
