from typing import Dict, Optional

import mlflow


class ModelRegistry:
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = "sentiment-analysis"
        mlflow.set_experiment(self.experiment_name)

    def register_model(self, model, metrics: Dict, tags: Optional[Dict] = None):
        """Register model with MLflow"""
        with mlflow.start_run():
            # log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            # log model
            mlflow.pytorch.log_model(model, "model", registered_model_name="sentiment-model")

            run_id = mlflow.active_run().info.run_id

        return run_id

    def load_model(self, version: str = "latest"):
        """Load model from registry"""
        if version == "latest":
            model = mlflow.pytorch.load_model("models:/sentiment-model/Production")
        else:
            model = mlflow.pytorch.load_model(f"models:/sentiment-model/{version}")
        return model

    def promote_model(self, run_id: str, stage: str = "Production"):
        """Promote model to production"""
        client = mlflow.tracking.MLflowClient()
        model_version = client.cerate_model_version(
            name="sentiment-model", source=f"runs:/{run_id}/model", run_id=run_id
        )
        client.transition_model_version_stage(
            name="sentiment-model", version=model_version.version, stage=stage
        )
