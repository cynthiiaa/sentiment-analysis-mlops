#!/usr/bin/env python3
"""
Model evaluation script for sentiment analysis models.
Evaluates model performance on test datasets and checks against thresholds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_path: str) -> Tuple:
    """Load model and tokenizer from path"""
    print(f"Loading model from {model_path}")

    # Try to load from local path first
    if Path(model_path).exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Fallback to HuggingFace hub
        print(f"Local model not found, loading from HuggingFace: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Set to eval mode
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def evaluate_on_dataset(
    model,
    tokenizer,
    device,
    dataset_name: str = "glue",
    task: str = "sst2",
    split: str = "validation",
) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
    """Evaluate model on a specific dataset"""
    print(f"\nEvaluating on {dataset_name}/{task} - {split} split")

    # Load dataset
    dataset = load_dataset(dataset_name, task, split=split)

    # Prepare predictions
    all_predictions = []
    all_labels = []

    # Process in batches
    batch_size = 32
    total_samples = len(dataset)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch = dataset[i : min(i + batch_size, total_samples)]

            # Tokenize
            inputs = tokenizer(
                batch["sentence"],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            # Predict
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"])

            # Progress
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, total_samples)}/{total_samples} samples")

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_score": f1_score(all_labels, all_predictions, average="weighted"),
        "precision": precision_score(all_labels, all_predictions, average="weighted"),
        "recall": recall_score(all_labels, all_predictions, average="weighted"),
    }

    return metrics, all_predictions, all_labels


def check_performance_threshold(metrics: Dict[str, float], threshold: float) -> bool:
    """Check if model meets performance threshold"""
    # Use F1 score as the main metric
    return metrics["f1_score"] >= threshold


def save_evaluation_results(
    metrics: Dict[str, float],
    predictions: List[int],
    labels: List[int],
    output_path: str = "results/evaluation_results.json",
):
    """Save evaluation results to file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    results = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "total_samples": int(len(predictions)),
        "correct_predictions": int(sum(p == l for p, l in zip(predictions, labels))),
        "class_distribution": {
            "predicted": {
                "positive": int(sum(1 for p in predictions if p == 1)),
                "negative": int(sum(1 for p in predictions if p == 0)),
            },
            "actual": {
                "positive": int(sum(1 for label in labels if label == 1)),
                "negative": int(sum(1 for label in labels if label == 0)),
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_evaluation_summary(metrics: Dict[str, float]):
    """Print evaluation summary"""
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():15s}: {value:.4f}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/latest",
        help="Path to model directory or HuggingFace model name",
    )
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset name (default: glue)")
    parser.add_argument(
        "--task", type=str, default="sst2", help="Task name for GLUE dataset (default: sst2)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate on (default: validation)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Performance threshold for F1 score (default: 0.85)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    try:
        # Load model
        model, tokenizer, device = load_model_and_tokenizer(args.model_path)

        # Evaluate
        metrics, predictions, labels = evaluate_on_dataset(
            model, tokenizer, device, dataset_name=args.dataset, task=args.task, split=args.split
        )

        # Print summary
        print_evaluation_summary(metrics)

        # Save results
        save_evaluation_results(metrics, predictions, labels, args.output)

        # Check threshold
        if check_performance_threshold(metrics, args.threshold):
            print(
                f"\n✅ Model PASSED: F1 score ({metrics['f1_score']:.4f}) >= threshold ({args.threshold})"
            )
            return 0
        else:
            print(
                f"\n❌ Model FAILED: F1 score ({metrics['f1_score']:.4f}) < threshold ({args.threshold})"
            )
            return 1

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
