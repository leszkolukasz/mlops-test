"""
Model training module with Ray Tune for hyperparameter optimization.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import os
import json


class ModelTrainer:
    """
    Model trainer with hyperparameter tuning support.
    """
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train, params=None):
        """
        Train a Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: Hyperparameters for the model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        self.best_params = params
        
        return self.model
    
    def evaluate(self, X, y):
        """
        Evaluate the model on given data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
        # Save parameters
        params_path = path.replace('.pkl', '_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = joblib.load(path)
        
        # Load parameters if available
        params_path = path.replace('.pkl', '_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.best_params = json.load(f)


def train_model_with_config(config, X_train, y_train, X_val, y_val):
    """
    Training function for Ray Tune hyperparameter optimization.
    
    Args:
        config: Dictionary with hyperparameters
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    """
    from ray import train as ray_train
    
    trainer = ModelTrainer()
    
    # Train with given config
    params = {
        'n_estimators': config['n_estimators'],
        'max_depth': config['max_depth'],
        'min_samples_split': config['min_samples_split'],
        'min_samples_leaf': config['min_samples_leaf'],
        'random_state': 42
    }
    
    trainer.train(X_train, y_train, params)
    
    # Evaluate on validation set
    metrics = trainer.evaluate(X_val, y_val)
    
    # Report metrics to Ray Tune
    ray_train.report(metrics)


if __name__ == "__main__":
    # Test the model trainer
    from feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.process_pipeline()
    
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    metrics = trainer.evaluate(X_test, y_test)
    print("Test metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")
