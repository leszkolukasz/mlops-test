"""Model predictor with Ray Serve for scalable inference."""

import ray
from ray import serve
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import mlflow
import mlflow.sklearn
import pickle
import time
from pathlib import Path

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import ServingConfig
from mlops_pipeline.training.model_registry import ModelRegistry


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1}
)
class ModelPredictor(LoggerMixin):
    """Ray Serve deployment for model predictions."""
    
    def __init__(self, config: ServingConfig, model_name: str, model_version: str = "latest"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.scaler = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load model and associated artifacts."""
        self.log_info(
            "Loading model for serving",
            model_name=self.model_name,
            model_version=self.model_version
        )
        
        try:
            # Load model from registry
            model_registry = ModelRegistry(config=None)  # Will use default config
            self.model, self.metadata = model_registry.get_model(
                self.model_name,
                self.model_version
            )
            
            # Try to load feature columns and scaler if available
            # In a real deployment, these would be stored alongside the model
            data_path = Path("data/processed")
            
            if (data_path / "feature_columns.pkl").exists():
                with open(data_path / "feature_columns.pkl", "rb") as f:
                    self.feature_columns = pickle.load(f)
            
            if (data_path / "scaler.pkl").exists():
                with open(data_path / "scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            
            self.log_info(
                "Model loaded successfully",
                model_version=self.metadata.get("version"),
                model_stage=self.metadata.get("stage"),
                feature_columns=len(self.feature_columns) if self.feature_columns else 0
            )
            
        except Exception as e:
            self.log_error(
                "Failed to load model",
                error=str(e),
                model_name=self.model_name,
                model_version=self.model_version
            )
            raise
    
    def preprocess_input(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Select and order feature columns if available
        if self.feature_columns:
            # Add missing columns with default values
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Select only required columns in correct order
            df = df[self.feature_columns]
        
        # Apply scaling if available
        if self.scaler and self.feature_columns:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df
    
    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Make predictions on input data."""
        start_time = time.time()
        
        try:
            # Preprocess input
            df = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # Get prediction probabilities if available (classification)
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilities = self.model.predict_proba(df)
                except Exception:
                    pass  # Some models might not support probabilities
            
            # Prepare response
            response = {
                "predictions": predictions.tolist(),
                "model_name": self.model_name,
                "model_version": self.metadata.get("version", self.model_version),
                "prediction_time_ms": (time.time() - start_time) * 1000,
                "num_samples": len(df)
            }
            
            if probabilities is not None:
                response["probabilities"] = probabilities.tolist()
                # Add class names if binary classification
                if probabilities.shape[1] == 2:
                    response["classes"] = ["class_0", "class_1"]
            
            self.log_info(
                "Prediction completed",
                num_samples=len(df),
                prediction_time_ms=response["prediction_time_ms"]
            )
            
            return response
            
        except Exception as e:
            self.log_error(
                "Prediction failed",
                error=str(e),
                data_type=type(data).__name__
            )
            return {
                "error": str(e),
                "model_name": self.model_name,
                "model_version": self.model_version
            }
    
    def batch_predict(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        return self.predict(batch_data)
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_name": self.model_name,
            "model_version": self.metadata.get("version", self.model_version) if self.metadata else self.model_version,
            "model_loaded": self.model is not None,
            "timestamp": time.time()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.metadata:
            return {"error": "Model metadata not available"}
        
        return {
            "model_name": self.model_name,
            "version": self.metadata.get("version"),
            "stage": self.metadata.get("stage"),
            "run_id": self.metadata.get("run_id"),
            "metrics": self.metadata.get("metrics", {}),
            "parameters": self.metadata.get("params", {}),
            "feature_columns": self.feature_columns,
            "creation_timestamp": self.metadata.get("creation_timestamp"),
            "last_updated_timestamp": self.metadata.get("last_updated_timestamp")
        }
    
    async def __call__(self, request) -> Dict[str, Any]:
        """Handle HTTP requests."""
        return self.predict(request)


def create_predictor_deployment(
    model_name: str,
    model_version: str = "latest",
    num_replicas: int = 2,
    max_concurrent_queries: int = 100
) -> serve.deployment:
    """Create a model predictor deployment with custom configuration."""
    
    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": 1}
    )
    class CustomModelPredictor(ModelPredictor):
        def __init__(self):
            from mlops_pipeline.utils.config import ServingConfig
            config = ServingConfig()
            super().__init__(config, model_name, model_version)
    
    return CustomModelPredictor