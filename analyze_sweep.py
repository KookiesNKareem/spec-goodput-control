#!/usr/bin/env python3
"""Summarize benchmark sweep logs into a per-batch comparison table."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from auto_quant_spec_controller import iter_aggregate_rows


def condition_label(row: dict[str, Any]) -> str:
    condition = row["condition"]
    if condition == "target":
        return "target"
    gamma = row.get("spec_tokens")
    draft = row.get("draft") or ""
    if condition == "spec_ngram":
        return f"ngram g{gamma}"
    short = draft.split("/")[-1]
    short = short.replace("Qwen2.5-", "").replace("-Instruct", "")
    return f"{short} g{gamma}"


def steady_tok_s(row: dict[str, Any]) -> float:
    return float(row.get("output_tok_s_steady_mean", row["output_tok_s_mean"]))


def steady_acceptance(row: dict[str, Any]) -> float | None:
    for key in ("draft_acceptance_rate_steady_mean", "draft_acceptance_rate_mean"):
        if key in row:
            return float(row[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--min-gain", type=float, default=0.03)
    args = parser.parse_args()

    by_batch: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    labels: list[str] = []
    for row in iter_aggregate_rows(args.logs):
        label = condition_label(row)
        if label not in labels:
            labels.append(label)
        by_batch[int(row["num_prompts"])][label] = row

    if "target" in labels:
        labels.remove("target")
        labels.insert(0, "target")

    header = ["batch"]
    for label in labels:
        header.append(f"{label} tok/s")
        if label != "target":
            header.append(f"{label} speedup")
            header.append(f"{label} acc")
    header.append("best action")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")

    for batch in sorted(by_batch):
        rows = by_batch[batch]
        target_row = rows.get("target")
        target_tps = steady_tok_s(target_row) if target_row else None
        cells = [str(batch)]
        best_label, best_tps = "off", target_tps or 0.0
        for label in labels:
            row = rows.get(label)
            if row is None:
                cells.append("-")
                if label != "target":
                    cells.extend(["-", "-"])
                continue
            tps = steady_tok_s(row)
            cells.append(f"{tps:.1f}")
            if label != "target":
                if target_tps:
                    speedup = tps / target_tps - 1.0
                    cells.append(f"{speedup:+.1%}")
                    if tps > best_tps * (1.0 + args.min_gain):
                        best_label, best_tps = label, tps
                else:
                    cells.append("-")
                acceptance = steady_acceptance(row)
                cells.append(f"{acceptance:.3f}" if acceptance is not None else "-")
        cells.append(f"{best_label} ({best_tps:.1f})")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
