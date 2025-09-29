"""Hyperparameter tuning with Ray Tune."""

import pandas as pd
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import mlflow
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import TrainingConfig


def tune_objective(config: Dict[str, Any]) -> None:
    """Objective function for hyperparameter tuning."""
    # Get data and configuration
    X_train = config["X_train"]
    y_train = config["y_train"]
    X_val = config["X_val"]
    y_val = config["y_val"]
    model_type = config["model_type"]
    task_type = config["task_type"]
    
    # Extract hyperparameters
    hyperparams = {k: v for k, v in config.items() 
                   if k not in ["X_train", "y_train", "X_val", "y_val", "model_type", "task_type"]}
    
    # Initialize model
    try:
        if task_type == "classification":
            if model_type == "random_forest":
                model = RandomForestClassifier(**hyperparams, random_state=42, n_jobs=1)
            elif model_type == "logistic_regression":
                model = LogisticRegression(**hyperparams, random_state=42, max_iter=1000)
            elif model_type == "svm":
                model = SVC(**hyperparams, random_state=42, probability=True)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        else:  # regression
            if model_type == "random_forest":
                model = RandomForestRegressor(**hyperparams, random_state=42, n_jobs=1)
            elif model_type == "linear_regression":
                model = LinearRegression(**hyperparams)
            elif model_type == "svm":
                model = SVR(**hyperparams)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metric to optimize
        if task_type == "classification":
            # Use F1 score as primary metric for classification
            score = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            accuracy = accuracy_score(y_val, y_pred)
            tune.report(f1_score=score, accuracy=accuracy)
        else:
            # Use negative MSE for regression (to maximize)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            tune.report(mse=mse, r2=r2, neg_mse=-mse)
            
    except Exception as e:
        # Report poor performance for invalid hyperparameters
        if task_type == "classification":
            tune.report(f1_score=0.0, accuracy=0.0)
        else:
            tune.report(mse=float('inf'), r2=-float('inf'), neg_mse=-float('inf'))


class HyperparameterTuner(LoggerMixin):
    """Hyperparameter tuner using Ray Tune."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
    def get_search_space(self, model_type: str, task_type: str) -> Dict[str, Any]:
        """Get hyperparameter search space for different models."""
        if task_type == "classification":
            if model_type == "random_forest":
                return {
                    "n_estimators": tune.choice([50, 100, 200, 300]),
                    "max_depth": tune.choice([3, 5, 10, 15, None]),
                    "min_samples_split": tune.choice([2, 5, 10]),
                    "min_samples_leaf": tune.choice([1, 2, 4]),
                    "max_features": tune.choice(["sqrt", "log2", None])
                }
            elif model_type == "logistic_regression":
                return {
                    "C": tune.loguniform(0.01, 100),
                    "penalty": tune.choice(["l1", "l2", "elasticnet"]),
                    "solver": tune.choice(["liblinear", "saga"]),
                    "l1_ratio": tune.uniform(0, 1)  # Only used with elasticnet
                }
            elif model_type == "svm":
                return {
                    "C": tune.loguniform(0.1, 100),
                    "kernel": tune.choice(["rbf", "poly", "sigmoid"]),
                    "gamma": tune.choice(["scale", "auto"] + list(tune.loguniform(0.001, 1).sample(3)))
                }
        else:  # regression
            if model_type == "random_forest":
                return {
                    "n_estimators": tune.choice([50, 100, 200, 300]),
                    "max_depth": tune.choice([3, 5, 10, 15, None]),
                    "min_samples_split": tune.choice([2, 5, 10]),
                    "min_samples_leaf": tune.choice([1, 2, 4]),
                    "max_features": tune.choice(["sqrt", "log2", None])
                }
            elif model_type == "linear_regression":
                return {
                    "fit_intercept": tune.choice([True, False]),
                    "positive": tune.choice([True, False])
                }
            elif model_type == "svm":
                return {
                    "C": tune.loguniform(0.1, 100),
                    "kernel": tune.choice(["rbf", "poly", "sigmoid"]),
                    "gamma": tune.choice(["scale", "auto"] + list(tune.loguniform(0.001, 1).sample(3))),
                    "epsilon": tune.loguniform(0.01, 1)
                }
        
        return {}
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: Optional[str] = None,
        max_trials: Optional[int] = None,
        time_budget: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Tune hyperparameters using Ray Tune."""
        model_type = model_type or self.config.model_type
        max_trials = max_trials or self.config.max_trials
        
        # Determine task type
        task_type = "classification" if y_train.nunique() < 20 else "regression"
        
        self.log_info(
            "Starting hyperparameter tuning",
            model_type=model_type,
            task_type=task_type,
            max_trials=max_trials,
            train_samples=len(X_train),
            val_samples=len(X_val)
        )
        
        # Get search space
        search_space = self.get_search_space(model_type, task_type)
        
        if not search_space:
            self.log_warning(f"No search space defined for {model_type}")
            return {}, {}
        
        # Add data and model info to search space
        search_space.update({
            "X_train": X_train.values,
            "y_train": y_train.values,
            "X_val": X_val.values,
            "y_val": y_val.values,
            "model_type": model_type,
            "task_type": task_type
        })
        
        # Configure search algorithm
        search_alg = HyperOptSearch(metric="f1_score" if task_type == "classification" else "neg_mse", mode="max")
        
        # Configure scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="f1_score" if task_type == "classification" else "neg_mse",
            mode="max",
            max_t=100,
            grace_period=10,
            reduction_factor=2
        )
        
        # Run hyperparameter tuning
        tuner = tune.Tuner(
            tune.with_resources(tune_objective, resources={"cpu": 1}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=max_trials,
                time_budget_s=time_budget
            ),
            run_config=ray.air.RunConfig(
                name=f"hyperparameter_tuning_{model_type}",
                local_dir="./ray_results"
            )
        )
        
        results = tuner.fit()
        
        # Get best trial
        best_result = results.get_best_result(
            metric="f1_score" if task_type == "classification" else "neg_mse",
            mode="max"
        )
        
        # Extract best hyperparameters (exclude data)
        best_hyperparams = {
            k: v for k, v in best_result.config.items()
            if k not in ["X_train", "y_train", "X_val", "y_val", "model_type", "task_type"]
        }
        
        best_metrics = best_result.metrics
        
        self.log_info(
            "Hyperparameter tuning completed",
            best_hyperparameters=best_hyperparams,
            best_metrics=best_metrics
        )
        
        return best_hyperparams, best_metrics
    
    def tune_with_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: Optional[str] = None,
        cv_folds: int = 5,
        max_trials: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Tune hyperparameters using cross-validation."""
        model_type = model_type or self.config.model_type
        max_trials = max_trials or self.config.max_trials
        
        # Determine task type
        task_type = "classification" if y.nunique() < 20 else "regression"
        
        self.log_info(
            "Starting hyperparameter tuning with cross-validation",
            model_type=model_type,
            task_type=task_type,
            cv_folds=cv_folds,
            max_trials=max_trials,
            samples=len(X)
        )
        
        # Get search space (excluding data parameters)
        base_search_space = self.get_search_space(model_type, task_type)
        
        if not base_search_space:
            self.log_warning(f"No search space defined for {model_type}")
            return {}, {}
        
        # Custom objective function for cross-validation
        def cv_objective(config: Dict[str, Any]) -> None:
            try:
                # Initialize model with hyperparameters
                if task_type == "classification":
                    if model_type == "random_forest":
                        model = RandomForestClassifier(**config, random_state=42, n_jobs=1)
                        scoring = "f1_weighted"
                    elif model_type == "logistic_regression":
                        model = LogisticRegression(**config, random_state=42, max_iter=1000)
                        scoring = "f1_weighted"
                    elif model_type == "svm":
                        model = SVC(**config, random_state=42, probability=True)
                        scoring = "f1_weighted"
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                else:  # regression
                    if model_type == "random_forest":
                        model = RandomForestRegressor(**config, random_state=42, n_jobs=1)
                        scoring = "neg_mean_squared_error"
                    elif model_type == "linear_regression":
                        model = LinearRegression(**config)
                        scoring = "neg_mean_squared_error"
                    elif model_type == "svm":
                        model = SVR(**config)
                        scoring = "neg_mean_squared_error"
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=cv_folds, 
                    scoring=scoring,
                    n_jobs=1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                tune.report(
                    cv_score=mean_score,
                    cv_std=std_score,
                    cv_scores=cv_scores.tolist()
                )
                
            except Exception as e:
                # Report poor performance for invalid hyperparameters
                tune.report(cv_score=-float('inf'), cv_std=float('inf'))
        
        # Run hyperparameter tuning
        tuner = tune.Tuner(
            tune.with_resources(cv_objective, resources={"cpu": 1}),
            param_space=base_search_space,
            tune_config=tune.TuneConfig(
                search_alg=HyperOptSearch(metric="cv_score", mode="max"),
                scheduler=ASHAScheduler(
                    metric="cv_score",
                    mode="max",
                    max_t=100,
                    grace_period=10,
                    reduction_factor=2
                ),
                num_samples=max_trials
            ),
            run_config=ray.air.RunConfig(
                name=f"cv_hyperparameter_tuning_{model_type}",
                local_dir="./ray_results"
            )
        )
        
        results = tuner.fit()
        
        # Get best trial
        best_result = results.get_best_result(metric="cv_score", mode="max")
        
        best_hyperparams = best_result.config
        best_metrics = best_result.metrics
        
        self.log_info(
            "Cross-validation hyperparameter tuning completed",
            best_hyperparameters=best_hyperparams,
            best_cv_score=best_metrics.get("cv_score"),
            best_cv_std=best_metrics.get("cv_std")
        )
        
        return best_hyperparams, best_metrics
    
    def get_default_hyperparameters(self, model_type: str, task_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        if task_type == "classification":
            if model_type == "random_forest":
                return {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt"
                }
            elif model_type == "logistic_regression":
                return {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "liblinear"
                }
            elif model_type == "svm":
                return {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale"
                }
        else:  # regression
            if model_type == "random_forest":
                return {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt"
                }
            elif model_type == "linear_regression":
                return {
                    "fit_intercept": True,
                    "positive": False
                }
            elif model_type == "svm":
                return {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale",
                    "epsilon": 0.1
                }
        
        return {}