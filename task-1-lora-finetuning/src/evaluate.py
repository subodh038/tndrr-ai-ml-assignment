"""
evaluate.py
-----------
Evaluation utilities for the NL→JSON calendar assistant model.

Usage:
    python evaluate.py --test dataset/test.json --predictions results/predictions_finetuned.json
                       --output results/metrics.json
"""
import json
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple


# ── Core evaluation functions ─────────────────────────────────────────────────

def is_valid_json(text: str) -> bool:
    """
    Check whether a string is valid JSON.

    Args:
        text: String to validate.

    Returns:
        True if `text` can be parsed as JSON, False otherwise.
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def field_exact_match(pred_json: Dict[str, Any], gold_json: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compare fields of a predicted JSON dict against a gold (ground truth) JSON dict.

    For each key in `gold_json`, the function checks whether the predicted
    dict contains that key with the exact same value (case-insensitive string
    comparison for string values).

    Args:
        pred_json: Predicted JSON as a Python dict.
        gold_json: Ground-truth JSON as a Python dict.

    Returns:
        Dict mapping each gold field name to a bool (True = exact match).
    """
    result = {}
    for key, gold_val in gold_json.items():
        pred_val = pred_json.get(key)
        if isinstance(gold_val, str) and isinstance(pred_val, str):
            result[key] = gold_val.strip().lower() == pred_val.strip().lower()
        else:
            result[key] = gold_val == pred_val
    return result


def compute_metrics(
    predictions: List[str],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics over a list of model predictions vs gold outputs.

    Metrics returned:
        - json_validity_rate (float): fraction of predictions that are valid JSON
        - field_exact_match_per_field (dict[str, float]): per-field accuracy
        - overall_exact_match (float): fraction where ALL gold fields matched
        - action_accuracy (float): accuracy on the 'action' field specifically
        - num_valid_json (int)
        - total (int)

    Args:
        predictions:   List of raw prediction strings (model output text).
        ground_truth:  List of gold output dicts (the 'output' field of each example).

    Returns:
        Metrics dict.
    """
    assert len(predictions) == len(ground_truth), "Length mismatch"

    n = len(predictions)
    n_valid = 0
    field_match_counts: Dict[str, int] = {}
    field_total_counts: Dict[str, int] = {}
    n_overall_exact = 0

    for pred_str, gold in zip(predictions, ground_truth):
        if is_valid_json(pred_str):
            n_valid += 1
            pred = json.loads(pred_str)
            if isinstance(pred, dict):
                matches = field_exact_match(pred, gold)
                all_match = all(matches.values())
                if all_match:
                    n_overall_exact += 1
                for field, matched in matches.items():
                    field_match_counts[field] = field_match_counts.get(field, 0) + (1 if matched else 0)
                    field_total_counts[field] = field_total_counts.get(field, 0) + 1

    # Per-field accuracy
    field_accuracy = {}
    for field in field_total_counts:
        field_accuracy[field] = round(
            field_match_counts.get(field, 0) / field_total_counts[field], 4
        )

    return {
        "total": n,
        "num_valid_json": n_valid,
        "json_validity_rate": round(n_valid / n, 4) if n > 0 else 0.0,
        "field_exact_match_per_field": field_accuracy,
        "overall_exact_match": round(n_overall_exact / n, 4) if n > 0 else 0.0,
        "action_accuracy": field_accuracy.get("action", 0.0),
    }


def find_failure_cases(
    predictions: List[str],
    ground_truth: List[Dict[str, Any]],
    inputs: Optional[List[str]] = None,
    n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find the first `n` prediction failures and diagnose the cause.

    Failure causes:
        - "invalid_json": prediction is not parseable JSON
        - "action_mismatch": action field differs
        - "field_mismatch": one or more non-action fields differ
        - "missing_fields": predicted JSON is missing required gold fields

    Args:
        predictions:  List of prediction strings.
        ground_truth: List of gold output dicts.
        inputs:       (Optional) list of input strings for context.
        n:            Number of failure cases to return.

    Returns:
        List of dicts describing each failure case.
    """
    failures = []
    for idx, (pred_str, gold) in enumerate(zip(predictions, ground_truth)):
        if len(failures) >= n:
            break

        entry: Dict[str, Any] = {
            "index": idx,
            "input": inputs[idx] if inputs else None,
            "gold": gold,
            "prediction_raw": pred_str,
        }

        if not is_valid_json(pred_str):
            entry["cause"] = "invalid_json"
            entry["detail"] = "Model output could not be parsed as JSON"
            failures.append(entry)
            continue

        pred = json.loads(pred_str)
        if not isinstance(pred, dict):
            entry["cause"] = "invalid_json"
            entry["detail"] = "Parsed JSON is not an object/dict"
            failures.append(entry)
            continue

        # Check for missing gold fields
        missing = [k for k in gold if k not in pred]
        if missing:
            entry["cause"] = "missing_fields"
            entry["detail"] = f"Missing fields: {missing}"
            entry["prediction_parsed"] = pred
            failures.append(entry)
            continue

        matches = field_exact_match(pred, gold)
        if all(matches.values()):
            continue  # not a failure

        if "action" in matches and not matches["action"]:
            entry["cause"] = "action_mismatch"
            entry["detail"] = f"Gold action={gold.get('action')!r}, pred action={pred.get('action')!r}"
        else:
            mismatched = [f for f, ok in matches.items() if not ok]
            entry["cause"] = "field_mismatch"
            entry["detail"] = f"Mismatched fields: {mismatched}"

        entry["prediction_parsed"] = pred
        entry["field_matches"] = matches
        failures.append(entry)

    return failures


def save_results(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics dict to a JSON file.

    Args:
        metrics: Metrics dict from `compute_metrics`.
        path:    Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate NL→JSON model predictions")
    parser.add_argument("--test", required=True, help="Path to test.json")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON (list of strings)")
    parser.add_argument("--output", default="results/metrics.json", help="Output metrics path")
    parser.add_argument("--failures", type=int, default=3, help="Number of failure cases to report")
    args = parser.parse_args()

    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(args.predictions, "r", encoding="utf-8") as f:
        raw_predictions = json.load(f)

    # raw_predictions may be a list of strings or list of dicts with 'prediction' key
    if raw_predictions and isinstance(raw_predictions[0], dict):
        preds = [p.get("prediction", "") for p in raw_predictions]
        inputs = [p.get("input", None) for p in raw_predictions]
    else:
        preds = [str(p) for p in raw_predictions]
        inputs = [ex["input"] for ex in test_data]

    gold_outputs = [ex["output"] for ex in test_data]

    if len(preds) != len(gold_outputs):
        raise ValueError(
            f"Prediction count ({len(preds)}) != test example count ({len(gold_outputs)})"
        )

    metrics = compute_metrics(preds, gold_outputs)
    save_results(metrics, args.output)

    print("\n=== Evaluation Results ===")
    print(json.dumps(metrics, indent=2))

    failures = find_failure_cases(preds, gold_outputs, inputs=inputs, n=args.failures)
    if failures:
        print(f"\n=== First {len(failures)} Failure Cases ===")
        for fc in failures:
            print(json.dumps(fc, indent=2, default=str))


if __name__ == "__main__":
    main()
