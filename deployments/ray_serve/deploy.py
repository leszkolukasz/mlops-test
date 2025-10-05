"""
Ray Serve deployment for Iris classifier.
"""
import ray
from ray import serve
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

app = FastAPI()


class IrisFeatures(BaseModel):
    """Input features for iris prediction"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int
    prediction_label: str
    probabilities: List[float]


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5})
@serve.ingress(app)
class IrisClassifier:
    """Ray Serve deployment for Iris classification"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the classifier with model and scaler.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.class_names = ['setosa', 'versicolor', 'virginica']
        
    def _create_features(self, sepal_length, sepal_width, petal_length, petal_width):
        """Create engineered features"""
        # Original features
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Engineered features
        petal_area = petal_length * petal_width
        sepal_area = sepal_length * sepal_width
        petal_sepal_length_ratio = petal_length / (sepal_length + 1e-8)
        petal_sepal_width_ratio = petal_width / (sepal_width + 1e-8)
        
        features.extend([petal_area, sepal_area, petal_sepal_length_ratio, petal_sepal_width_ratio])
        
        return np.array(features).reshape(1, -1)
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(self, features: IrisFeatures) -> PredictionResponse:
        """
        Make a prediction for iris species.
        
        Args:
            features: Input features (sepal and petal measurements)
            
        Returns:
            Prediction with class label and probabilities
        """
        # Create features with engineering
        X = self._create_features(
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0].tolist()
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=self.class_names[prediction],
            probabilities=probabilities
        )
    
    @app.get("/health")
    async def health(self):
        """Health check endpoint"""
        return {"status": "healthy"}


def deploy_model(model_path: str = "/app/models/iris_model.pkl", 
                scaler_path: str = "/app/models/scaler.pkl",
                host: str = "0.0.0.0",
                port: int = 8000):
    """
    Deploy the model using Ray Serve.
    
    Args:
        model_path: Path to the trained model
        scaler_path: Path to the scaler
        host: Host to bind to
        port: Port to bind to
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Start Ray Serve
    serve.start(http_options={"host": host, "port": port})
    
    # Deploy the classifier
    IrisClassifier.bind(model_path, scaler_path)
    
    print(f"Model deployed successfully at http://{host}:{port}")
    print("Endpoints:")
    print(f"  - POST http://{host}:{port}/predict")
    print(f"  - GET  http://{host}:{port}/health")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Iris classifier with Ray Serve")
    parser.add_argument("--model-path", default="../../models/iris_model.pkl", 
                       help="Path to model file")
    parser.add_argument("--scaler-path", default="../../models/scaler.pkl",
                       help="Path to scaler file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    deploy_model(args.model_path, args.scaler_path, args.host, args.port)
    
    # Keep the service running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
