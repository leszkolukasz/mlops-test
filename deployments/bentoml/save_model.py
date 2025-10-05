"""
BentoML service for Iris classifier.
"""
import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray
import joblib
import os


class IrisBentoService:
    """BentoML service for Iris classification"""
    
    def __init__(self):
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


# Load model and scaler
def load_model_and_scaler(model_path="../../models/iris_model.pkl", 
                          scaler_path="../../models/scaler.pkl"):
    """Load model and scaler"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Create BentoML service
def create_service(model_path="../../models/iris_model.pkl",
                   scaler_path="../../models/scaler.pkl"):
    """
    Create and save BentoML service.
    
    Args:
        model_path: Path to the trained model
        scaler_path: Path to the scaler
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # Save to BentoML model store
    bentoml.sklearn.save_model(
        "iris_classifier",
        model,
        signatures={
            "predict": {"batchable": True},
            "predict_proba": {"batchable": True}
        },
        custom_objects={
            "scaler": scaler
        },
        metadata={
            "model_type": "RandomForestClassifier",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "engineered_features": ["petal_area", "sepal_area", 
                                   "petal_sepal_length_ratio", "petal_sepal_width_ratio"],
            "classes": ["setosa", "versicolor", "virginica"]
        }
    )
    
    print("Model saved to BentoML model store")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BentoML service for Iris classifier")
    parser.add_argument("--model-path", default="../../models/iris_model.pkl",
                       help="Path to model file")
    parser.add_argument("--scaler-path", default="../../models/scaler.pkl",
                       help="Path to scaler file")
    
    args = parser.parse_args()
    
    create_service(args.model_path, args.scaler_path)
