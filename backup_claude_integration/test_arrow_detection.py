#!/usr/bin/env python3
"""Run Hough connector matching on the clean flowchart fixtures with ground-truth verification."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DEFAULT_WEIGHTS, build_labeled_demo_whiteboard, load_classifier, run_pipeline

DEFAULT_DEBUG = PROJECT_ROOT / "data" / "debug"
FIXTURES = [
    PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_a.jpg",
    PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_b.jpg",
    PROJECT_ROOT / "data" / "validation" / "val_flowchart_1.png",
    PROJECT_ROOT / "data" / "validation" / "val_flowchart_2.png",
    PROJECT_ROOT / "data" / "validation" / "val_flowchart_3.png",
]

# Ground-truth verified directed edge lists (source -> target)
GROUND_TRUTH_EDGES = {
    "synthetic_demo": [
        (0, 1),
        (2, 3),
        (3, 4),
    ],
    "flowchart_clean_a": [
        (0, 2),
        (1, 5),
        (2, 6),
        (3, 5),
        (4, 5),
        (7, 13),
        (8, 9),
        (8, 10),
        (10, 14),
        (12, 13),
    ],
    "flowchart_clean_b": [
        (0, 1),
        (0, 4),
        (1, 4),
        (2, 3),
        (3, 5),
        (6, 11),
        (6, 12),
        (7, 11),
        (8, 9),
        (8, 10),
        (11, 12),
    ],
    "val_flowchart_1": [
        (3, 2),
        (2, 1),
        (1, 0),
    ],
    "val_flowchart_2": [
        (2, 4),
        (4, 1),
        (1, 3),
        (4, 0),
    ],
    "val_flowchart_3": [
        (2, 3),
        (2, 0),
        (3, 1),
        (0, 1),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arrow/Hough relationship test.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG)
    return parser.parse_args()


def evaluate_edges(detected_edges: list[tuple[int, int]], expected_edges: list[tuple[int, int]]) -> dict:
    det_dir = set(detected_edges)
    exp_dir = set(expected_edges)

    det_undir = set(tuple(sorted(e)) for e in detected_edges)
    exp_undir = set(tuple(sorted(e)) for e in expected_edges)

    tp_dir = len(det_dir & exp_dir)
    fp_dir = len(det_dir - exp_dir)
    fn_dir = len(exp_dir - det_dir)

    precision_dir = tp_dir / (tp_dir + fp_dir) if (tp_dir + fp_dir) > 0 else 0.0
    recall_dir = tp_dir / (tp_dir + fn_dir) if (tp_dir + fn_dir) > 0 else 0.0
    f1_dir = (2 * precision_dir * recall_dir) / (precision_dir + recall_dir) if (precision_dir + recall_dir) > 0 else 0.0

    tp_undir = len(det_undir & exp_undir)
    direction_accuracy = tp_dir / tp_undir if tp_undir > 0 else 1.0

    return {
        "expected": len(exp_dir),
        "detected": len(det_dir),
        "tp_dir": tp_dir,
        "fp_dir": fp_dir,
        "fn_dir": fn_dir,
        "precision_dir": precision_dir,
        "recall_dir": recall_dir,
        "f1_dir": f1_dir,
        "direction_accuracy": direction_accuracy,
        "fp_edges": sorted(list(det_dir - exp_dir)),
        "fn_edges": sorted(list(exp_dir - det_dir)),
    }


def run_one(image, name: str, model, device, debug_dir: Path) -> dict | None:
    out_dir = debug_dir / name
    result = run_pipeline(
        image,
        model=model,
        device=device,
        skip_ocr=True,
        debug_dir=out_dir,
    )
    print(f"\n=== {name} ===")
    for line in result.log_lines:
        if line.startswith("  pairing"):
            continue
        print(" ", line)
    print(
        f" nodes={result.n_nodes}  edges={result.n_edges}  "
        f"contour_arrows={result.n_arrows_classified}  hough={result.n_hough_lines}"
    )
    print(f" debug={result.debug_image_path}")
    g = result.graph
    det_edges = list(g.edges())
    print(" directed edges:")
    if not det_edges:
        print("   (none)")
    for u, v in sorted(det_edges):
        la = g.nodes[u].get("label", "")
        lb = g.nodes[v].get("label", "")
        print(
            f"   {u}:{g.nodes[u].get('shape_type')}({la!r}) -> "
            f"{v}:{g.nodes[v].get('shape_type')}({lb!r})"
        )

    if name in GROUND_TRUTH_EDGES:
        exp_edges = GROUND_TRUTH_EDGES[name]
        metrics = evaluate_edges(det_edges, exp_edges)
        print(" Evaluation Metrics (Directed):")
        print(
            f"   Expected: {metrics['expected']}  Detected: {metrics['detected']}  "
            f"TP: {metrics['tp_dir']}  FP: {metrics['fp_dir']}  FN: {metrics['fn_dir']}"
        )
        print(
            f"   Directed Precision: {metrics['precision_dir']:.3f}  "
            f"Directed Recall: {metrics['recall_dir']:.3f}  "
            f"Directed F1: {metrics['f1_dir']:.3f}  "
            f"Direction Accuracy: {metrics['direction_accuracy']:.3f}"
        )
        if metrics["fn_edges"]:
            print(f"   Missed Directed Edges (FN): {metrics['fn_edges']}")
        if metrics["fp_edges"]:
            print(f"   False Positive Directed Edges (FP): {metrics['fp_edges']}")
        return metrics
    return None


def main() -> int:
    args = parse_args()
    if not DEFAULT_WEIGHTS.is_file():
        print(f"Missing {DEFAULT_WEIGHTS}", file=sys.stderr)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_classifier(DEFAULT_WEIGHTS, device)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            return 1
        run_one(image, args.image.stem, model, device, args.debug_dir)
        return 0

    all_metrics = []
    m = run_one(build_labeled_demo_whiteboard(), "synthetic_demo", model, device, args.debug_dir)
    if m:
        all_metrics.append(m)
    for path in FIXTURES:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        m = run_one(image, path.stem, model, device, args.debug_dir)
        if m:
            all_metrics.append(m)

    if all_metrics:
        total_exp = sum(m["expected"] for m in all_metrics)
        total_det = sum(m["detected"] for m in all_metrics)
        total_tp = sum(m["tp_dir"] for m in all_metrics)
        total_fp = sum(m["fp_dir"] for m in all_metrics)
        total_fn = sum(m["fn_dir"] for m in all_metrics)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        dir_acc = total_tp / sum(m["tp_dir"] + m["fp_dir"] for m in all_metrics) if total_det > 0 else 1.0
        print("\n================ BENCHMARK SUMMARY ================")
        print(f"Fixtures Evaluated: {len(all_metrics)}")
        print(f"Total Expected Edges: {total_exp}")
        print(f"Total Detected Edges: {total_det}")
        print(f"Total TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
        print(f"Micro-Averaged Directed Precision: {micro_p:.3f}")
        print(f"Micro-Averaged Directed Recall:    {micro_r:.3f}")
        print(f"Micro-Averaged Directed F1-Score:  {micro_f1:.3f}")
        print(f"Overall Direction Accuracy:        {dir_acc:.3f}")
        print("===================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())



if __name__ == "__main__":
    sys.exit(main())
