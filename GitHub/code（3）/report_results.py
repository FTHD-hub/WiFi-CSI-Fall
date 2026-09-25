import argparse
import json
from pathlib import Path

from train import normalize_scenario_name


PAPER_RESULTS = {
    "scenario1": {
        "title": "Scenario 1: Aggregated Dataset",
        "image_heading": "Paper Figure Snapshot",
        "tables": [
            {
                "title": "Paper Table: Aggregated Dataset",
                "columns": ["Method", "Acc", "Prec", "Rec"],
                "rows": [
                    ["Modified B0 [34]", "94.9%", "91.7%", "96.8%"],
                    ["FallDeFi", "80.8%", "75.0%", "81.9%"],
                    ["Work in [28]", "88.0%", "82.9%", "90.2%"],
                    ["Proposed", "99.5%", "98.8%", "100.0%"],
                ],
            }
        ],
    },
    "scenario2": {
        "title": "Scenario 2: NLoS Dataset (C_NLoS)",
        "tables": [
            {
                "title": "Paper Table: Right Living Room C / NLoS",
                "columns": ["Method", "Acc", "Prec", "Rec"],
                "rows": [
                    ["Modified B0 [34]", "82.0%", "72.7%", "100.0%"],
                    ["FallDeFi", "68.2%", "66.4%", "70.9%"],
                    ["Work in [28]", "75.4%", "72.7%", "76.2%"],
                    ["Proposed", "97.6%", "97.3%", "100.0%"],
                ],
            }
        ],
    },
    "scenario2paper": {
        "title": "Scenario 2 Paper: NLoS Dataset (C_NLoS)",
        "tables": [
            {
                "title": "Paper Table: Right Living Room C / NLoS",
                "columns": ["Method", "Acc", "Prec", "Rec"],
                "rows": [
                    ["Modified B0 [34]", "82.0%", "72.7%", "100.0%"],
                    ["FallDeFi", "68.2%", "66.4%", "70.9%"],
                    ["Work in [28]", "75.4%", "72.7%", "76.2%"],
                    ["Proposed", "97.6%", "97.3%", "100.0%"],
                ],
            }
        ],
    },
    "scenario2heldout": {
        "title": "Scenario 2 Held-Out: NLoS Dataset (C_NLoS)",
        "tables": [
            {
                "title": "Paper Table: Right Living Room C / NLoS",
                "columns": ["Method", "Acc", "Prec", "Rec"],
                "rows": [
                    ["Modified B0 [34]", "82.0%", "72.7%", "100.0%"],
                    ["FallDeFi", "68.2%", "66.4%", "70.9%"],
                    ["Work in [28]", "75.4%", "72.7%", "76.2%"],
                    ["Proposed", "97.6%", "97.3%", "100.0%"],
                ],
            }
        ],
    },
    "scenario3": {
        "title": "Scenario 3: Unseen Room A and D",
        "tables": [
            {
                "title": "Paper Figure: Lecture Room D",
                "columns": ["Method", "Acc", "Prec", "Rec", "F1"],
                "rows": [
                    ["Modified B0 [34]", "72.0%", "69.0%", "77.0%", "73.0%"],
                    ["FallDeFi", "68.0%", "66.0%", "72.0%", "69.0%"],
                    ["Work in [28]", "73.0%", "72.0%", "77.0%", "74.0%"],
                    ["Proposed", "92.0%", "94.0%", "90.0%", "92.0%"],
                ],
            },
            {
                "title": "Paper Figure: Living Room A",
                "columns": ["Method", "Acc", "Prec", "Rec", "F1"],
                "rows": [
                    ["Modified B0 [34]", "69.0%", "55.0%", "90.0%", "68.0%"],
                    ["FallDeFi", "60.0%", "49.0%", "79.0%", "61.0%"],
                    ["Work in [28]", "66.0%", "53.0%", "94.0%", "68.0%"],
                    ["Proposed", "88.0%", "82.0%", "87.0%", "84.0%"],
                ],
            },
        ],
    },
    "scenario4": {
        "title": "Scenario 4: Sparse Training Data",
        "tables": [
            {
                "title": "Paper Figure: Left Living Room C / LoS",
                "columns": ["Method", "Acc", "Prec", "Rec", "F1"],
                "rows": [
                    ["Modified B0 [34]", "90.0%", "92.0%", "88.0%", "90.0%"],
                    ["FallDeFi", "73.0%", "69.0%", "80.0%", "74.0%"],
                    ["Work in [28]", "76.0%", "74.0%", "83.0%", "78.0%"],
                    ["Proposed", "99.0%", "98.0%", "100.0%", "99.0%"],
                ],
            },
            {
                "title": "Paper Figure: Right Living Room C / NLoS",
                "columns": ["Method", "Acc", "Prec", "Rec", "F1"],
                "rows": [
                    ["Modified B0 [34]", "78.0%", "66.0%", "95.0%", "78.0%"],
                    ["FallDeFi", "68.0%", "62.0%", "76.0%", "68.0%"],
                    ["Work in [28]", "73.0%", "68.0%", "85.0%", "76.0%"],
                    ["Proposed", "93.0%", "92.0%", "92.0%", "92.0%"],
                ],
            },
        ],
    },
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_pct(value):
    return f"{value * 100:.2f}%"


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_report_from_counts(metrics):
    tp = int(metrics.get("val_tp", 0))
    tn = int(metrics.get("val_tn", 0))
    fp = int(metrics.get("val_fp", 0))
    fn = int(metrics.get("val_fn", 0))
    total = int(metrics.get("val_count", tp + tn + fp + fn))

    fall_support = tp + fn
    nonfall_support = tn + fp

    fall_prec = safe_div(tp, tp + fp)
    fall_rec = safe_div(tp, tp + fn)
    fall_f1 = safe_div(2 * fall_prec * fall_rec, fall_prec + fall_rec)

    nonfall_prec = safe_div(tn, tn + fn)
    nonfall_rec = safe_div(tn, tn + fp)
    nonfall_f1 = safe_div(2 * nonfall_prec * nonfall_rec, nonfall_prec + nonfall_rec)

    accuracy = safe_div(tp + tn, total)
    macro_prec = (fall_prec + nonfall_prec) / 2
    macro_rec = (fall_rec + nonfall_rec) / 2
    macro_f1 = (fall_f1 + nonfall_f1) / 2
    weighted_prec = safe_div(fall_prec * fall_support + nonfall_prec * nonfall_support, total)
    weighted_rec = safe_div(fall_rec * fall_support + nonfall_rec * nonfall_support, total)
    weighted_f1 = safe_div(fall_f1 * fall_support + nonfall_f1 * nonfall_support, total)

    split_row = {
        "accuracy": accuracy,
        "f1": weighted_f1,
        "samples": total,
    }
    report_rows = [
        ("Fall", fall_prec, fall_rec, fall_f1, fall_support),
        ("Nonfall", nonfall_prec, nonfall_rec, nonfall_f1, nonfall_support),
        ("Macro Avg", macro_prec, macro_rec, macro_f1, total),
        ("Weighted Avg", weighted_prec, weighted_rec, weighted_f1, total),
    ]
    return split_row, report_rows


def markdown_table(columns, rows):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider] + body)


def build_reference_section(scenario, image_path=None):
    paper = PAPER_RESULTS.get(scenario, {})
    lines = [f"# {paper.get('title', 'Paper Reference')}"]
    if image_path and scenario == "scenario1":
        lines.extend(["", "## Paper Snapshot", f"![paper snapshot]({image_path})"])
    for table in paper.get("tables", []):
        lines.extend(["", f"## {table['title']}", markdown_table(table["columns"], table["rows"])])
    return "\n".join(lines)


def build_repro_section(train_payload=None, test_payload=None):
    lines = ["", "# Reproduction Results"]
    if train_payload:
        best = train_payload.get("best", {})
        lines.extend(
            [
                "",
                "## Best Validation Result",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["Accuracy", to_pct(best.get("val_acc", 0.0))],
                        ["Precision", to_pct(best.get("val_prec", 0.0))],
                        ["Recall", to_pct(best.get("val_rec", 0.0))],
                        ["F1-Score", to_pct(best.get("val_f1", 0.0))],
                        ["Samples", str(best.get("val_count", 0))],
                    ],
                ),
            ]
        )
    if test_payload:
        split_row, report_rows = classification_report_from_counts(test_payload["metrics"])
        lines.extend(
            [
                "",
                "## Test Split Summary",
                markdown_table(
                    ["Split", "Accuracy", "F1-Score", "Samples"],
                    [[
                        "test",
                        to_pct(split_row["accuracy"]),
                        to_pct(split_row["f1"]),
                        str(split_row["samples"]),
                    ]],
                ),
                "",
                "## Classification Report (test)",
                markdown_table(
                    ["Class", "Precision", "Recall", "F1-Score", "Support"],
                    [
                        [name, to_pct(prec), to_pct(rec), to_pct(f1), str(support)]
                        for name, prec, rec, f1, support in report_rows
                    ],
                ),
            ]
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a markdown experiment report from train/test JSON metrics.")
    parser.add_argument("--scenario", required=True, help="Scenario name, for example scenario1 or scenario2.")
    parser.add_argument("--train-json", type=Path, help="Path to the training metrics JSON.")
    parser.add_argument("--test-json", type=Path, help="Path to the test metrics JSON.")
    parser.add_argument("--output", type=Path, default=Path("results/experiment_report.md"), help="Output markdown path.")
    parser.add_argument("--image-path", type=str, help="Optional image path to embed for the paper reference snapshot.")
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = f"scenario{normalize_scenario_name(args.scenario)}"
    train_payload = load_json(args.train_json) if args.train_json else None
    test_payload = load_json(args.test_json) if args.test_json else None

    reference = build_reference_section(scenario, image_path=args.image_path)
    reproduction = build_repro_section(train_payload=train_payload, test_payload=test_payload)
    content = reference + "\n" + reproduction + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
