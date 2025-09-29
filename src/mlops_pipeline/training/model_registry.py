"""Model registry for managing trained models with MLflow."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
import pickle
import json

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import TrainingConfig


class ModelRegistry(LoggerMixin):
    """Model registry for managing ML models lifecycle."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.client = None
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow client and tracking."""
        mlflow.set_tracking_uri(self.config.model_registry_uri)
        self.client = MlflowClient(tracking_uri=self.config.model_registry_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.client.create_experiment("mlops-pipeline-models")
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        model_version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a model in the registry."""
        self.log_info(
            "Registering model",
            model_name=model_name,
            model_version=model_version
        )
        
        with mlflow.start_run(experiment_id=self.client.get_experiment_by_name("mlops-pipeline-models").experiment_id):
            # Log hyperparameters
            if hyperparameters:
                mlflow.log_params(hyperparameters)
            
            # Log metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
            
            # Log metadata as tags
            if metadata:
                for key, value in metadata.items():
                    mlflow.set_tag(key, str(value))
            
            # Log custom tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log the model
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            
            run_id = mlflow.active_run().info.run_id
            
        self.log_info(
            "Model registered successfully",
            model_name=model_name,
            run_id=run_id,
            model_uri=model_info.model_uri
        )
        
        return run_id
    
    def get_model(
        self,
        model_name: str,
        version: str = "latest",
        stage: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Retrieve a model from the registry."""
        self.log_info(
            "Retrieving model",
            model_name=model_name,
            version=version,
            stage=stage
        )
        
        try:
            if stage:
                # Get model by stage
                model_version = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )[0]
                model_uri = f"models:/{model_name}/{stage}"
            elif version == "latest":
                # Get latest version
                model_versions = self.client.get_latest_versions(model_name)
                if not model_versions:
                    raise ValueError(f"No versions found for model {model_name}")
                model_version = model_versions[0]
                model_uri = f"models:/{model_name}/latest"
            else:
                # Get specific version
                model_version = self.client.get_model_version(model_name, version)
                model_uri = f"models:/{model_name}/{version}"
            
            # Load the model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get model metadata
            run = self.client.get_run(model_version.run_id)
            metadata = {
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp
            }
            
            self.log_info(
                "Model retrieved successfully",
                model_name=model_name,
                version=model_version.version,
                stage=model_version.current_stage
            )
            
            return model, metadata
            
        except Exception as e:
            self.log_error(
                "Failed to retrieve model",
                model_name=model_name,
                version=version,
                stage=stage,
                error=str(e)
            )
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        self.log_info("Listing all registered models")
        
        models = []
        for model in self.client.search_registered_models():
            model_info = {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "latest_versions": []
            }
            
            # Get latest versions for each stage
            for stage in ["None", "Staging", "Production", "Archived"]:
                try:
                    latest_versions = self.client.get_latest_versions(
                        model.name, stages=[stage]
                    )
                    for version in latest_versions:
                        version_info = {
                            "version": version.version,
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                            "creation_timestamp": version.creation_timestamp
                        }
                        model_info["latest_versions"].append(version_info)
                except Exception:
                    continue
            
            models.append(model_info)
        
        self.log_info("Models listed successfully", total_models=len(models))
        return models
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> None:
        """Promote a model to a specific stage."""
        self.log_info(
            "Promoting model",
            model_name=model_name,
            version=version,
            stage=stage,
            archive_existing=archive_existing
        )
        
        # Archive existing model in target stage if requested
        if archive_existing and stage in ["Staging", "Production"]:
            try:
                existing_versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                for existing_version in existing_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=existing_version.version,
                        stage="Archived"
                    )
                    self.log_info(
                        "Archived existing model version",
                        version=existing_version.version,
                        from_stage=stage
                    )
            except Exception as e:
                self.log_warning(
                    "Failed to archive existing model",
                    error=str(e)
                )
        
        # Promote the new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        self.log_info(
            "Model promoted successfully",
            model_name=model_name,
            version=version,
            stage=stage
        )
    
    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version."""
        self.log_info(
            "Deleting model version",
            model_name=model_name,
            version=version
        )
        
        self.client.delete_model_version(model_name, version)
        
        self.log_info(
            "Model version deleted successfully",
            model_name=model_name,
            version=version
        )
    
    def delete_registered_model(self, model_name: str) -> None:
        """Delete an entire registered model."""
        self.log_info("Deleting registered model", model_name=model_name)
        
        self.client.delete_registered_model(model_name)
        
        self.log_info("Registered model deleted successfully", model_name=model_name)
    
    def update_model_description(
        self,
        model_name: str,
        description: str,
        version: Optional[str] = None
    ) -> None:
        """Update model or model version description."""
        if version:
            self.log_info(
                "Updating model version description",
                model_name=model_name,
                version=version
            )
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        else:
            self.log_info(
                "Updating model description",
                model_name=model_name
            )
            self.client.update_registered_model(
                name=model_name,
                description=description
            )
    
    def get_model_metrics(
        self,
        model_name: str,
        version: str = "latest"
    ) -> Dict[str, float]:
        """Get metrics for a specific model version."""
        try:
            if version == "latest":
                model_versions = self.client.get_latest_versions(model_name)
                if not model_versions:
                    return {}
                model_version = model_versions[0]
            else:
                model_version = self.client.get_model_version(model_name, version)
            
            run = self.client.get_run(model_version.run_id)
            return run.data.metrics
            
        except Exception as e:
            self.log_error(
                "Failed to get model metrics",
                model_name=model_name,
                version=version,
                error=str(e)
            )
            return {}
    
    def compare_models(
        self,
        model_specs: List[Tuple[str, str]],  # List of (model_name, version) tuples
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple models by their metrics."""
        self.log_info(
            "Comparing models",
            model_count=len(model_specs),
            metrics=metrics
        )
        
        comparison_data = []
        
        for model_name, version in model_specs:
            try:
                model_metrics = self.get_model_metrics(model_name, version)
                
                row = {
                    "model_name": model_name,
                    "version": version
                }
                
                if metrics:
                    for metric in metrics:
                        row[metric] = model_metrics.get(metric, None)
                else:
                    row.update(model_metrics)
                
                comparison_data.append(row)
                
            except Exception as e:
                self.log_warning(
                    "Failed to get metrics for model",
                    model_name=model_name,
                    version=version,
                    error=str(e)
                )
        
        comparison_df = pd.DataFrame(comparison_data)
        
        self.log_info(
            "Model comparison completed",
            models_compared=len(comparison_data)
        )
        
        return comparison_df
    
    def search_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for models based on filter criteria."""
        self.log_info(
            "Searching models",
            filter_string=filter_string,
            max_results=max_results
        )
        
        models = self.client.search_registered_models(
            filter_string=filter_string,
            max_results=max_results
        )
        
        model_list = []
        for model in models:
            model_info = {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "tags": model.tags if hasattr(model, 'tags') else {}
            }
            model_list.append(model_info)
        
        self.log_info(
            "Model search completed",
            models_found=len(model_list)
        )
        
        return model_list