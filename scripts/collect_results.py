#!/usr/bin/env python3
"""Collect benchmark JSON records from ignored log files.

The benchmark scripts print line-delimited JSON with stable prefixes:

* RESULTS_AGG_JSON from fixed-batch decode sweeps.
* ARRIVAL_RESULT_JSON from open-loop arrival benchmarks.
* V1_CHECK_JSON from current-vLLM off-state spot checks.

This collector keeps the raw records auditable while producing a compact
Markdown index for public repo artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PREFIXES = {
    "RESULTS_AGG_JSON ": "sweeps",
    "ARRIVAL_RESULT_JSON ": "arrivals",
    "V1_CHECK_JSON ": "v1_checks",
}


def read_records(paths: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {
        "sweeps": [],
        "arrivals": [],
        "v1_checks": [],
    }
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        with path.open(errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                for prefix, bucket in PREFIXES.items():
                    if line.startswith(prefix):
                        row = json.loads(line[len(prefix):])
                        row["log_path"] = str(path)
                        row["log_line"] = line_number
                        records[bucket].append(row)
                        break
    return records


def tok_s(row: dict[str, Any]) -> float:
    return float(row.get("output_tok_s_steady_mean", row["output_tok_s_mean"]))


def acceptance(row: dict[str, Any]) -> str:
    for key in ("draft_acceptance_rate_steady_mean",
                "draft_acceptance_rate_mean"):
        if key in row:
            return f"{float(row[key]):.3f}"
    return "-"


def short_model(name: str | None) -> str:
    if not name:
        return "-"
    return name.split("/")[-1].replace("-Instruct", "")


def action_label(row: dict[str, Any]) -> str:
    condition = row.get("condition")
    if condition == "target":
        return "target"
    gamma = row.get("runtime_spec_tokens", row.get("spec_tokens"))
    if condition == "spec_ngram":
        return f"ngram g{gamma}"
    if condition == "spec_awq":
        return f"{short_model(row.get('draft'))} g{gamma}"
    return str(condition)


def sweep_group_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        short_model(row.get("target")),
        str(row.get("prompt_source", "builtin")),
        int(row.get("max_tokens", 0)),
        str(row.get("temperature", "0.0")),
    )


def render_sweeps(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    lines = ["## Fixed-batch Sweeps", ""]
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[sweep_group_key(row)].append(row)

    for key in sorted(groups):
        target, prompt_source, max_tokens, temperature = key
        group = groups[key]
        labels = []
        by_batch: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            label = action_label(row)
            if label not in labels:
                labels.append(label)
            by_batch[int(row["num_prompts"])][label] = row
        if "target" in labels:
            labels.remove("target")
            labels.insert(0, "target")

        lines.append(
            f"### {target}, prompts={prompt_source}, max_tokens={max_tokens}, "
            f"temperature={temperature}"
        )
        lines.append("")
        header = ["batch"]
        for label in labels:
            header.append(f"{label} tok/s")
            if label != "target":
                header.extend([f"{label} speedup", f"{label} acc"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")

        for batch in sorted(by_batch):
            rows_by_label = by_batch[batch]
            target_row = rows_by_label.get("target")
            target_tok_s = tok_s(target_row) if target_row else None
            cells = [str(batch)]
            for label in labels:
                row = rows_by_label.get(label)
                if row is None:
                    cells.append("-")
                    if label != "target":
                        cells.extend(["-", "-"])
                    continue
                value = tok_s(row)
                cells.append(f"{value:.1f}")
                if label != "target":
                    if target_tok_s:
                        cells.append(f"{value / target_tok_s - 1.0:+.1%}")
                    else:
                        cells.append("-")
                    cells.append(acceptance(row))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def render_arrivals(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    lines = ["## Arrival-process Runs", ""]
    lines.append(
        "| qps | condition | tok/s | e2e p50 | e2e p90 | e2e p99 | "
        "ttft p50 | max running |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (float(r["qps"]), str(r["condition"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['qps']):g}",
                    str(row["condition"]),
                    f"{float(row['output_tok_s']):.1f}",
                    f"{float(row['e2e_p50_s']):.2f}",
                    f"{float(row['e2e_p90_s']):.2f}",
                    f"{float(row['e2e_p99_s']):.2f}",
                    f"{float(row['ttft_p50_s']):.3f}",
                    str(row["max_observed_running"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_v1(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    lines = ["## V1 Off-state Checks", ""]
    lines.append(
        "| tag | condition | vllm | prompts | eager | lookup | spec tokens | steady tok/s |"
    )
    lines.append("|---|---|---|---:|---|---|---:|---:|")
    for row in sorted(rows, key=lambda r: (str(r.get("tag", "")), str(r["condition"]))):
        lookup = row.get("prompt_lookup")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("tag") or "-"),
                    str(row["condition"]),
                    str(row.get("vllm_version", "-")),
                    str(row["num_prompts"]),
                    str(bool(row.get("enforce_eager"))).lower(),
                    str(lookup if lookup is not None else "-"),
                    str(row.get("spec_tokens") or "-"),
                    f"{float(row['tok_s_steady_mean']):.1f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_markdown(records: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Collected Results",
        "",
        "Generated from local ignored log files. See each JSON record's "
        "`log_path` and `log_line` fields for provenance.",
        "",
        "Record counts:",
        "",
        f"- Fixed-batch sweeps: {len(records['sweeps'])}",
        f"- Arrival-process runs: {len(records['arrivals'])}",
        f"- V1 off-state checks: {len(records['v1_checks'])}",
        "",
    ]
    lines.extend(render_sweeps(records["sweeps"]))
    lines.extend(render_arrivals(records["arrivals"]))
    lines.extend(render_v1(records["v1_checks"]))
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log files to scan.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.logs)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    markdown = render_markdown(records)
    if args.markdown_out:
        path = Path(args.markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
