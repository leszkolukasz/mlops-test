"""Test configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from mlops_pipeline.utils.config import Config, DataConfig, TrainingConfig


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    
    assert config.data.batch_size == 1000
    assert config.data.train_test_split == 0.8
    assert config.training.model_type == "random_forest"
    assert config.serving.port == 8000
    assert config.monitoring.log_level == "INFO"


def test_config_from_yaml():
    """Test loading configuration from YAML file."""
    config_data = {
        "data": {
            "batch_size": 500,
            "train_test_split": 0.75
        },
        "training": {
            "model_type": "logistic_regression",
            "max_trials": 20
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = Config.from_yaml(temp_path)
        
        assert config.data.batch_size == 500
        assert config.data.train_test_split == 0.75
        assert config.training.model_type == "logistic_regression"
        assert config.training.max_trials == 20
        
    finally:
        Path(temp_path).unlink()


def test_config_save_yaml():
    """Test saving configuration to YAML file."""
    config = Config()
    config.data.batch_size = 2000
    config.training.model_type = "svm"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save_yaml(temp_path)
        
        # Load back and verify
        loaded_config = Config.from_yaml(temp_path)
        assert loaded_config.data.batch_size == 2000
        assert loaded_config.training.model_type == "svm"
        
    finally:
        Path(temp_path).unlink()


def test_data_config_validation():
    """Test data configuration validation."""
    # Valid configuration
    config = DataConfig(
        batch_size=1000,
        train_test_split=0.8,
        validation_split=0.2
    )
    assert config.batch_size == 1000
    assert config.train_test_split == 0.8
    
    # Test defaults
    config = DataConfig()
    assert config.random_state == 42


def test_training_config_validation():
    """Test training configuration validation."""
    config = TrainingConfig(
        model_type="random_forest",
        hyperparameters={"n_estimators": 100},
        max_trials=10
    )
    
    assert config.model_type == "random_forest"
    assert config.hyperparameters["n_estimators"] == 100
    assert config.max_trials == 10