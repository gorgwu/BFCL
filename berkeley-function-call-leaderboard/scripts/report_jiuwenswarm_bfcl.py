#!/usr/bin/env python3
"""Compile JiuwenSwarm BFCL run outputs into a human-readable text report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


MODEL_DIR = "jiuwenswarm-system1-FC"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def accuracy_for(score_dir: Path) -> tuple[int, int, dict[str, tuple[int, int]]]:
    correct = total = 0
    categories: dict[str, tuple[int, int]] = {}
    model_score_dir = score_dir / MODEL_DIR
    if not model_score_dir.exists():
        return correct, total, categories
    for path in sorted(model_score_dir.rglob("*_score.json")):
        data = read_json(path)
        if isinstance(data, dict):
            header = data
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            header = data[0]
        else:
            continue
        c = int(header.get("correct_count", 0))
        t = int(header.get("total_count", 0))
        category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        categories[category] = (c, t)
        correct += c
        total += t
    return correct, total, categories


def result_metrics(result_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    model_result_dir = result_dir / MODEL_DIR
    cases: list[dict[str, Any]] = []
    backend_counts: dict[str, int] = {}
    if not model_result_dir.exists():
        return cases, backend_counts
    for path in sorted(model_result_dir.rglob("*_result.json")):
        for item in read_jsonl(path):
            # BFCL flattens handler metadata alongside id/result when writing
            # JSONL, but accept a nested metadata object for other handlers too.
            metadata = item.get("metadata") or item
            if not isinstance(metadata, dict):
                metadata = {}
            jiuwen = metadata.get("jiuwenswarm") or {}
            if isinstance(jiuwen, dict) and jiuwen.get("tool_discovery_backend"):
                backend = str(jiuwen["tool_discovery_backend"])
                backend_counts[backend] = backend_counts.get(backend, 0) + 1

            totals = metadata.get("conversation_totals") or {}
            latency = number(totals.get("latency_seconds")) if isinstance(totals, dict) else None
            turns = metadata.get("turn_metrics")
            if latency is None and isinstance(turns, list):
                vals = [number(turn.get("latency_seconds")) for turn in turns if isinstance(turn, dict)]
                known = [v for v in vals if v is not None]
                latency = sum(known) if known else None
            if latency is None:
                raw_latency = metadata.get("latency")
                if isinstance(raw_latency, list):
                    def flatten(values: list[Any]) -> list[float]:
                        out: list[float] = []
                        for value in values:
                            if isinstance(value, list):
                                out.extend(flatten(value))
                            elif number(value) is not None:
                                out.append(float(value))
                        return out
                    values = flatten(raw_latency)
                    latency = sum(values) if values else None
                else:
                    latency = number(raw_latency)

            cases.append({
                "id": item.get("id", "unknown"),
                "latency": latency,
                "turns": turns if isinstance(turns, list) else [],
                "usage": metadata.get("conversation_model_usage") or {},
                "backend": jiuwen.get("tool_discovery_backend") if isinstance(jiuwen, dict) else None,
                "path": path,
            })
    return cases, backend_counts


def usage_totals(result_dir: Path) -> dict[str, Any]:
    path = result_dir / MODEL_DIR / "usage_summary.json"
    if not path.exists():
        return {}
    data = read_json(path)
    return data.get("usage", {}) if isinstance(data, dict) else {}


def report_run(label: str, result_dir: Path, score_dir: Path) -> tuple[list[str], bool]:
    lines = [f"{label}", "=" * len(label)]
    if not result_dir.exists():
        lines.append(f"Result folder not found: {result_dir}")
        return lines + [""], False

    correct, total, categories = accuracy_for(score_dir)
    cases, backend_counts = result_metrics(result_dir)
    latencies = [case["latency"] for case in cases if case["latency"] is not None]
    usage = usage_totals(result_dir)
    lines.append(f"Result folder: {result_dir}")
    lines.append(f"Score folder:  {score_dir}")
    lines.append(f"Cases with generated results: {len(cases)}")
    lines.append(f"Scored accuracy: {correct}/{total} ({100 * correct / total:.2f}%)" if total else "Scored accuracy: N/A (run evaluation to create score files)")
    if categories:
        lines.append("Accuracy by category:")
        for category, (cat_correct, cat_total) in sorted(categories.items()):
            pct = 100 * cat_correct / cat_total if cat_total else 0
            lines.append(f"  {category}: {cat_correct}/{cat_total} ({pct:.2f}%)")

    lines.append(f"Conversation latency: {len(latencies)}/{len(cases)} cases have timings")
    if latencies:
        lines.append(f"  Total: {sum(latencies):,.2f} s | mean: {mean(latencies):,.2f} s | median: {median(latencies):,.2f} s | p95: {sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)]:,.2f} s")
    if backend_counts:
        lines.append("Recorded discovery backend counts: " + ", ".join(f"{key}={value}" for key, value in sorted(backend_counts.items())))

    if usage:
        lines.append("Provider usage and reported cost:")
        for component in ("deepseek", "jev", "other_agent_models", "total"):
            item = usage.get(component)
            if not isinstance(item, dict):
                continue
            cost = item.get("cost_usd")
            cost_text = "N/A" if cost is None else f"${float(cost):,.6f}"
            complete = item.get("cost_complete")
            completeness = "complete" if complete else "INCOMPLETE/unknown"
            lines.append(
                f"  {component}: {int(item.get('input_tokens') or 0):,} input + "
                f"{int(item.get('output_tokens') or 0):,} output = {int(item.get('total_tokens') or 0):,} tokens; "
                f"cost {cost_text} ({completeness}); requests {int(item.get('request_count') or 0):,}; "
                f"models: {', '.join(item.get('models') or []) or 'N/A'}"
            )
    else:
        lines.append("Provider usage/cost summary not found; rerun generation or check per-case result metadata.")

    lines.append("Per-case latency and tokens:")
    if not cases:
        lines.append("  No generated result entries found.")
    for case in cases:
        usage_total = case["usage"].get("total", {}) if isinstance(case["usage"], dict) else {}
        tok = usage_total.get("total_tokens") if isinstance(usage_total, dict) else None
        lines.append(f"  {case['id']}: {fmt(case['latency'])} s, {fmt(tok, 0)} tokens, backend={case['backend'] or 'unknown'}")
        for turn in case["turns"]:
            if isinstance(turn, dict):
                lines.append(f"    turn {turn.get('turn', '?')}: {fmt(number(turn.get('latency_seconds')))} s, {fmt(number(turn.get('total_tokens')), 0)} tokens")
    lines.append("")
    return lines, bool(cases or total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-result-dir", type=Path, default=Path("result/jiuwenswarm-jev"))
    parser.add_argument("--bm25-result-dir", type=Path, default=Path("result/jiuwenswarm-bm25"))
    parser.add_argument("--jev-score-dir", type=Path)
    parser.add_argument("--bm25-score-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("jiuwenswarm_bfcl_report.txt"))
    args = parser.parse_args()

    jev_score = args.jev_score_dir or Path("score") / args.jev_result_dir.name
    bm25_score = args.bm25_score_dir or Path("score") / args.bm25_result_dir.name
    lines = [
        "JiuwenSwarm BFCL comparison report",
        "Generated from BFCL result and score files.",
        "Conversation latency is adapter-measured turn time; it excludes some benchmark setup, evaluation, and file-writing overhead.",
        "Reported costs come from model/provider usage metadata and may be incomplete; check the completeness label.",
        "",
    ]
    found_any = False
    for label, result_dir, score_dir in (
        ("JEV", args.jev_result_dir, jev_score),
        ("BM25", args.bm25_result_dir, bm25_score),
    ):
        section, found = report_run(label, result_dir, score_dir)
        lines.extend(section)
        found_any = found_any or found
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {args.output.resolve()}")
    if not found_any:
        print("No benchmark outputs were found yet. Check the result/score paths and run evaluation for accuracy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
