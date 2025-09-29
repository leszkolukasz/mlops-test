"""Configuration management for the MLOps pipeline."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RayConfig(BaseModel):
    """Ray cluster configuration."""
    
    num_cpus: Optional[int] = Field(default=None, description="Number of CPUs per worker")
    num_gpus: Optional[int] = Field(default=0, description="Number of GPUs per worker")
    memory: Optional[int] = Field(default=None, description="Memory per worker in bytes")
    object_store_memory: Optional[int] = Field(default=None, description="Object store memory")
    dashboard_host: str = Field(default="0.0.0.0", description="Ray dashboard host")
    dashboard_port: int = Field(default=8265, description="Ray dashboard port")


class DataConfig(BaseModel):
    """Data pipeline configuration."""
    
    batch_size: int = Field(default=1000, description="Batch size for data processing")
    train_test_split: float = Field(default=0.8, description="Train/test split ratio")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    data_path: str = Field(default="data/", description="Path to data directory")
    processed_data_path: str = Field(default="data/processed/", description="Path to processed data")


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    
    model_type: str = Field(default="random_forest", description="Type of model to train")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    cross_validation_folds: int = Field(default=5, description="Number of CV folds")
    max_trials: int = Field(default=10, description="Maximum number of hyperparameter trials")
    metric: str = Field(default="accuracy", description="Primary metric for optimization")
    model_registry_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI")


class ServingConfig(BaseModel):
    """Model serving configuration."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    num_replicas: int = Field(default=2, description="Number of serving replicas")
    max_concurrent_queries: int = Field(default=100, description="Max concurrent queries per replica")
    model_name: str = Field(default="mlops-model", description="Name of the model to serve")
    model_version: str = Field(default="latest", description="Version of the model to serve")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    drift_detection_threshold: float = Field(default=0.1, description="Model drift threshold")
    performance_threshold: float = Field(default=0.05, description="Performance degradation threshold")


class Config(BaseModel):
    """Main configuration class for the MLOps pipeline."""
    
    ray: RayConfig = Field(default_factory=RayConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Ray configuration
        if os.getenv("RAY_NUM_CPUS"):
            config_dict.setdefault("ray", {})["num_cpus"] = int(os.getenv("RAY_NUM_CPUS"))
        if os.getenv("RAY_NUM_GPUS"):
            config_dict.setdefault("ray", {})["num_gpus"] = int(os.getenv("RAY_NUM_GPUS"))
        
        # Training configuration
        if os.getenv("MODEL_TYPE"):
            config_dict.setdefault("training", {})["model_type"] = os.getenv("MODEL_TYPE")
        if os.getenv("MLFLOW_TRACKING_URI"):
            config_dict.setdefault("training", {})["model_registry_uri"] = os.getenv("MLFLOW_TRACKING_URI")
        
        # Serving configuration
        if os.getenv("SERVING_PORT"):
            config_dict.setdefault("serving", {})["port"] = int(os.getenv("SERVING_PORT"))
        if os.getenv("NUM_REPLICAS"):
            config_dict.setdefault("serving", {})["num_replicas"] = int(os.getenv("NUM_REPLICAS"))
        
        return cls(**config_dict)
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)