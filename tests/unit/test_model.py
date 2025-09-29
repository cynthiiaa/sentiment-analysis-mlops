import pytest

from src.models.sentiment_model import ModelConfig, SentimentModel


class TestSentimentModel:
    @pytest.fixture
    def model(self):
        config = ModelConfig()
        return SentimentModel(config)

    def test_model_initialization(self, model):
        assert model.model is not None
        assert model.tokenizer is not None

    def test_single_prediction(self, model):
        text = "This is a great product!"
        results = model.predict([text])

        assert len(results) == 1
        assert "sentiment" in results[0]
        assert "confidence" in results[0]
        assert results[0]["sentiment"] in ["positive", "negative"]
        assert 0 <= results[0]["confidence"] <= 1

    def test_batch_prediction(self, model):
        texts = ["I love this!", "This sucks eggs.", "Not sure how I feel about this"]
        results = model.predict(texts)

        assert len(results) == 3
        for result in results:
            assert "probabilities" in result
            assert (
                abs(result["probabilities"]["positive"] + result["probabilities"]["negative"] - 1.0)
                < 0.01
            )

    @pytest.mark.parametrize(
        "text,expected_sentiment",
        [("This is absolutely fantastic!", "positive"), ("Worst experience ever", "negative")],
    )
    def test_sentiment_accuracy(self, model, text, expected_sentiment):
        results = model.predict([text])
        assert results[0]["sentiment"] == expected_sentiment
