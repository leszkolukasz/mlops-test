"""Model training with Ray Train and distributed computing."""

import pandas as pd
import numpy as np
import ray
from ray import train
from ray.train import Checkpoint
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import pickle
import json
from pathlib import Path

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import TrainingConfig


def train_model_func(config: Dict[str, Any]) -> None:
    """Training function to be executed by Ray Train."""
    import os
    import tempfile
    
    # Get data from config
    X_train = config["X_train"]
    y_train = config["y_train"]
    X_val = config.get("X_val")
    y_val = config.get("y_val")
    model_type = config["model_type"]
    hyperparameters = config["hyperparameters"]
    task_type = config["task_type"]
    
    # Initialize model
    if task_type == "classification":
        if model_type == "random_forest":
            model = RandomForestClassifier(**hyperparameters, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**hyperparameters, random_state=42)
        elif model_type == "svm":
            model = SVC(**hyperparameters, random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported classification model: {model_type}")
    else:  # regression
        if model_type == "random_forest":
            model = RandomForestRegressor(**hyperparameters, random_state=42)
        elif model_type == "linear_regression":
            model = LinearRegression(**hyperparameters)
        elif model_type == "svm":
            model = SVR(**hyperparameters)
        else:
            raise ValueError(f"Unsupported regression model: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_predictions = model.predict(X_train)
    metrics = {"train_samples": len(X_train)}
    
    if task_type == "classification":
        metrics.update({
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "train_precision": precision_score(y_train, train_predictions, average="weighted", zero_division=0),
            "train_recall": recall_score(y_train, train_predictions, average="weighted", zero_division=0),
            "train_f1": f1_score(y_train, train_predictions, average="weighted", zero_division=0)
        })
        
        # Add AUC for binary classification
        if len(np.unique(y_train)) == 2:
            train_proba = model.predict_proba(X_train)[:, 1]
            metrics["train_auc"] = roc_auc_score(y_train, train_proba)
    else:
        metrics.update({
            "train_mse": mean_squared_error(y_train, train_predictions),
            "train_mae": mean_absolute_error(y_train, train_predictions),
            "train_r2": r2_score(y_train, train_predictions)
        })
    
    # Validation metrics if validation data is available
    if X_val is not None and y_val is not None:
        val_predictions = model.predict(X_val)
        metrics["val_samples"] = len(X_val)
        
        if task_type == "classification":
            metrics.update({
                "val_accuracy": accuracy_score(y_val, val_predictions),
                "val_precision": precision_score(y_val, val_predictions, average="weighted", zero_division=0),
                "val_recall": recall_score(y_val, val_predictions, average="weighted", zero_division=0),
                "val_f1": f1_score(y_val, val_predictions, average="weighted", zero_division=0)
            })
            
            if len(np.unique(y_val)) == 2:
                val_proba = model.predict_proba(X_val)[:, 1]
                metrics["val_auc"] = roc_auc_score(y_val, val_proba)
        else:
            metrics.update({
                "val_mse": mean_squared_error(y_val, val_predictions),
                "val_mae": mean_absolute_error(y_val, val_predictions),
                "val_r2": r2_score(y_val, val_predictions)
            })
    
    # Save model to temporary file for checkpointing
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Create checkpoint
        checkpoint = Checkpoint.from_directory(temp_dir)
        train.report(metrics, checkpoint=checkpoint)


class ModelTrainer(LoggerMixin):
    """Model trainer with Ray Train integration."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.mlflow_client = None
        
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config.model_registry_uri)
        mlflow.set_experiment("mlops-pipeline-training")
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_type: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a model using Ray Train."""
        model_type = model_type or self.config.model_type
        hyperparameters = hyperparameters or self.config.hyperparameters
        
        # Determine task type
        task_type = "classification" if y_train.nunique() < 20 else "regression"
        
        self.log_info(
            "Starting model training",
            model_type=model_type,
            task_type=task_type,
            train_samples=len(X_train),
            val_samples=len(X_val) if X_val is not None else 0,
            hyperparameters=hyperparameters
        )
        
        # Setup MLflow
        self._setup_mlflow()
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("task_type", task_type)
            mlflow.log_params(hyperparameters)
            mlflow.log_param("train_samples", len(X_train))
            if X_val is not None:
                mlflow.log_param("val_samples", len(X_val))
            
            # Prepare training configuration
            train_config = {
                "X_train": X_train.values,
                "y_train": y_train.values,
                "X_val": X_val.values if X_val is not None else None,
                "y_val": y_val.values if y_val is not None else None,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "task_type": task_type
            }
            
            # Create Ray Train trainer
            trainer = train.DataParallelTrainer(
                train_loop_per_worker=train_model_func,
                train_loop_config=train_config,
                scaling_config=train.ScalingConfig(num_workers=1, use_gpu=False)
            )
            
            # Train the model
            result = trainer.fit()
            
            # Get the best checkpoint and metrics
            best_checkpoint = result.checkpoint
            metrics = result.metrics
            
            # Load the trained model from checkpoint
            with best_checkpoint.as_directory() as checkpoint_dir:
                model_path = Path(checkpoint_dir) / "model.pkl"
                with open(model_path, "rb") as f:
                    trained_model = pickle.load(f)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                trained_model,
                "model",
                registered_model_name=f"mlops-pipeline-{model_type}",
                metadata={
                    "task_type": task_type,
                    "model_type": model_type,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val) if X_val is not None else 0
                }
            )
            
            # Get run ID for reference
            run_id = mlflow.active_run().info.run_id
            
        self.log_info(
            "Model training completed",
            run_id=run_id,
            metrics=metrics
        )
        
        return trained_model, metrics
    
    def cross_validate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation on the model."""
        model_type = model_type or self.config.model_type
        hyperparameters = hyperparameters or self.config.hyperparameters
        cv_folds = cv_folds or self.config.cross_validation_folds
        
        # Determine task type
        task_type = "classification" if y.nunique() < 20 else "regression"
        
        self.log_info(
            "Starting cross-validation",
            model_type=model_type,
            task_type=task_type,
            cv_folds=cv_folds,
            samples=len(X)
        )
        
        # Initialize model
        if task_type == "classification":
            if model_type == "random_forest":
                model = RandomForestClassifier(**hyperparameters, random_state=42)
            elif model_type == "logistic_regression":
                model = LogisticRegression(**hyperparameters, random_state=42)
            elif model_type == "svm":
                model = SVC(**hyperparameters, random_state=42, probability=True)
            else:
                raise ValueError(f"Unsupported classification model: {model_type}")
            
            scoring_metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        else:
            if model_type == "random_forest":
                model = RandomForestRegressor(**hyperparameters, random_state=42)
            elif model_type == "linear_regression":
                model = LinearRegression(**hyperparameters)
            elif model_type == "svm":
                model = SVR(**hyperparameters)
            else:
                raise ValueError(f"Unsupported regression model: {model_type}")
            
            scoring_metrics = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        
        # Perform cross-validation
        cv_results = {}
        for scoring in scoring_metrics:
            scores = cross_val_score(
                model, X, y, 
                cv=cv_folds, 
                scoring=scoring,
                n_jobs=-1
            )
            
            cv_results[f"{scoring}_scores"] = scores.tolist()
            cv_results[f"{scoring}_mean"] = float(scores.mean())
            cv_results[f"{scoring}_std"] = float(scores.std())
        
        cv_results.update({
            "model_type": model_type,
            "task_type": task_type,
            "cv_folds": cv_folds,
            "samples": len(X)
        })
        
        self.log_info(
            "Cross-validation completed",
            primary_metric_mean=cv_results.get(f"{scoring_metrics[0]}_mean"),
            primary_metric_std=cv_results.get(f"{scoring_metrics[0]}_std")
        )
        
        return cv_results
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate trained model on test data."""
        if task_type is None:
            task_type = "classification" if y_test.nunique() < 20 else "regression"
        
        self.log_info(
            "Evaluating model",
            task_type=task_type,
            test_samples=len(X_test)
        )
        
        # Make predictions
        predictions = model.predict(X_test)
        
        metrics = {"test_samples": len(X_test)}
        
        if task_type == "classification":
            metrics.update({
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "test_recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "test_f1": f1_score(y_test, predictions, average="weighted", zero_division=0)
            })
            
            # Add AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                test_proba = model.predict_proba(X_test)[:, 1]
                metrics["test_auc"] = roc_auc_score(y_test, test_proba)
        else:
            metrics.update({
                "test_mse": mean_squared_error(y_test, predictions),
                "test_mae": mean_absolute_error(y_test, predictions),
                "test_r2": r2_score(y_test, predictions)
            })
            
            # Add RMSE
            metrics["test_rmse"] = np.sqrt(metrics["test_mse"])
        
        self.log_info("Model evaluation completed", metrics=metrics)
        
        return metrics