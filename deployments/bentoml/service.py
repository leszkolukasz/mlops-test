"""
BentoML service definition for Iris classifier.
"""
import bentoml
import numpy as np
from bentoml.io import JSON


# Load the model from BentoML model store
iris_model = bentoml.sklearn.get("iris_classifier:latest")
model_runner = iris_model.to_runner()

# Create service
svc = bentoml.Service("iris_classifier", runners=[model_runner])


def create_features(sepal_length, sepal_width, petal_length, petal_width):
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


@svc.api(input=JSON(), output=JSON())
async def predict(input_data: dict) -> dict:
    """
    Make a prediction for iris species.
    
    Input format:
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }
    
    Output format:
    {
        "prediction": int,
        "prediction_label": str,
        "probabilities": [float, float, float]
    }
    """
    # Extract features
    sepal_length = input_data['sepal_length']
    sepal_width = input_data['sepal_width']
    petal_length = input_data['petal_length']
    petal_width = input_data['petal_width']
    
    # Create features with engineering
    X = create_features(sepal_length, sepal_width, petal_length, petal_width)
    
    # Get scaler from custom objects
    scaler = iris_model.custom_objects['scaler']
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = await model_runner.predict.async_run(X_scaled)
    probabilities = await model_runner.predict_proba.async_run(X_scaled)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    
    return {
        "prediction": int(prediction[0]),
        "prediction_label": class_names[int(prediction[0])],
        "probabilities": probabilities[0].tolist()
    }


@svc.api(input=JSON(), output=JSON())
async def health() -> dict:
    """Health check endpoint"""
    return {"status": "healthy"}
