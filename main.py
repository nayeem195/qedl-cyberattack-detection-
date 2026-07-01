"""
QEDL command-line entry point.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --ablate-qfs
    python main.py --config config.yaml --ablate-pqc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import yaml

# The PQC layer differentiates through a complex-valued statevector
# simulation; TensorFlow's autodiff emits a benign "casting complex128 to
# float32" warning during this process. It does not affect correctness
# (Pauli-Z expectation values are mathematically real), so it's silenced
# here for a cleaner console experience.
warnings.filterwarnings("ignore", message=".*casting an input of type complex128.*")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.preprocessing import load_dataset, preprocess
from src.train import run_qedl_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the QEDL hybrid quantum-classical pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ablate-qfs", action="store_true", help="Disable quantum feature selection (ablation)")
    parser.add_argument("--ablate-pqc", action="store_true", help="Disable the PQC front end (ablation)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)

    print(f"[1/3] Loading and preprocessing dataset from {cfg['data']['path']} ...")
    X, y = load_dataset(cfg["data"]["path"], cfg["data"]["target_column"])
    splits, meta = preprocess(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
    )
    cfg["model"]["num_classes"] = meta["num_classes"]
    print(f"    -> {meta['num_classes']} classes: {meta['class_names']}")

    print("[2/3] Running QEDL pipeline (QFS -> QDE/PQC -> hybrid classifier, 10-fold CV) ...")
    results = run_qedl_pipeline(
        splits,
        meta,
        cfg,
        use_qfs=not args.ablate_qfs,
        use_pqc=not args.ablate_pqc,
    )

    print("[3/3] Saving results ...")
    summary_path = os.path.join(cfg["output"]["results_dir"], "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, indent=2)
    print(f"    -> Summary written to {summary_path}")

    if results["model"] is not None:
        model_dir = cfg["output"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        weights_path = os.path.join(model_dir, "qedl_model.weights.h5")
        results["model"].save_weights(weights_path)
        print(f"    -> Model weights saved to {weights_path}")
        print("       (reload by rebuilding via build_hybrid_model(...) with the same")
        print("        config.yaml, then calling model.load_weights(...))")

    print("\n=== 10-Fold Cross-Validation Summary ===")
    for metric, stats in results["summary"].items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
