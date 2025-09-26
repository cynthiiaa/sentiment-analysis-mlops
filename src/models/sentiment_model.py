import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SentimentModel:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.load_model()

    def load_model(self):
        """Load model with error handling and fallback"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            ).to(self.config.device)
            self.model.eval()
            logger.info(f"Model loaded: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment with confidence scores"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        results = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            confidence = probabilities.max().item()
            prediction = probabilities.argmax().item()

            results.append(
                {
                    "text": text,
                    "sentiment": "positive" if prediction == 1 else "negative",
                    "confidence": confidence,
                    "probabilities": {
                        "negative": probabilities[0][0].item(),
                        "positive": probabilities[0][1].item(),
                    },
                }
            )

        return results

    def get_model_info(self) -> Dict:
        """Return model metadata for monitoring"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "max_length": self.config.max_length,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
