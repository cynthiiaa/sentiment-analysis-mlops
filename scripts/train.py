import argparse
from pathlib import Path

import mlflow
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.models.model_registry import ModelRegistry


def load_config(config_path: str):
    """Load training configuration"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def train_model(config: dict):
    """Main training function"""
    # load dataset
    dataset = load_dataset("glue", "sst2")

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    # tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding=True,
            truncation=True,
            max_length=config["model"]["max_length"],
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["base_model"], num_labels=2
    )

    # setup training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # start MLflow run
    registry = ModelRegistry()
    with mlflow.start_run(run_id=None):
        # log parameters
        mlflow.log_params(config["training"])

        # train model
        trainer.train()

        # evaluate
        eval_results = trainer.evaluate()

        # log metrics
        for key, value in eval_results.items():
            mlflow.log_metrics(key, value)

        # save model
        model_path = Path(config["training"]["output_dir"]) / "final_model"
        trainer.save_model(model_path)

        # register model
        # run_id = mlflow.active_run().info.run_id
        registry.register_model(model, metrics=eval_results, tags={"training_config": str(config)})

    print(f"Training completed. Model saved to {model_path}")
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = train_model(config)
    print(f"Final metrics: {results}")


if __name__ == "__main__":
    main()
