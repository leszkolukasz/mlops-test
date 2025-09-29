#!/usr/bin/env python3
"""
Complete MLOps Pipeline Example

This script demonstrates how to use all components of the MLOps pipeline:
1. Data generation and processing
2. Model training with hyperparameter tuning
3. Model evaluation and registry
4. Model serving
5. Monitoring and observability

Usage:
    python examples/full_pipeline_example.py
"""

import ray
from pathlib import Path
import pandas as pd
import time

from mlops_pipeline.utils.config import Config
from mlops_pipeline.utils.logger import configure_logging, get_logger
from mlops_pipeline.data.generator import DataGenerator
from mlops_pipeline.data.processor import DataProcessor
from mlops_pipeline.data.validator import DataValidator
from mlops_pipeline.training.trainer import ModelTrainer
from mlops_pipeline.training.hyperparameter_tuner import HyperparameterTuner
from mlops_pipeline.training.model_registry import ModelRegistry
from mlops_pipeline.serving.model_server import ModelServer
from mlops_pipeline.monitoring.metrics_collector import MetricsCollector


def main():
    """Run the complete MLOps pipeline example."""
    # Configure logging
    configure_logging(log_level="INFO", log_format="text")
    logger = get_logger("pipeline_example")
    
    logger.info("Starting MLOps Pipeline Example")
    
    # Load configuration
    config = Config()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            dashboard_host=config.ray.dashboard_host,
            dashboard_port=config.ray.dashboard_port,
            ignore_reinit_error=True
        )
        logger.info(f"Ray Dashboard: http://{config.ray.dashboard_host}:{config.ray.dashboard_port}")
    
    try:
        # Step 1: Data Generation
        logger.info("=" * 50)
        logger.info("Step 1: Generating synthetic dataset")
        logger.info("=" * 50)
        
        data_generator = DataGenerator(config.data)
        df = data_generator.generate_classification_dataset(
            n_samples=5000,
            n_features=15,
            n_workers=2
        )
        
        # Save raw data
        raw_data_path = Path("data/raw/example_dataset.parquet")
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_data_path, index=False)
        logger.info(f"Raw dataset saved: {raw_data_path}")
        
        # Step 2: Data Processing
        logger.info("=" * 50)
        logger.info("Step 2: Processing dataset")
        logger.info("=" * 50)
        
        processor = DataProcessor(config.data)
        processed_df = processor.process_dataset(df, n_workers=2)
        
        # Create train/test/validation splits
        splits = processor.create_train_test_split(processed_df)
        
        # Save processed data
        processed_dir = "data/processed"
        saved_files = processor.save_processed_data(splits, processed_dir)
        logger.info(f"Processed data saved to: {processed_dir}")
        
        # Step 3: Data Validation
        logger.info("=" * 50)
        logger.info("Step 3: Validating data quality")
        logger.info("=" * 50)
        
        validator = DataValidator(config.data)
        
        for split_name, split_df in splits.items():
            logger.info(f"Validating {split_name} split")
            
            # Quality validation
            quality_results = validator.validate_data_quality(split_df)
            logger.info(f"{split_name} quality score: {quality_results['quality_score']:.3f}")
            
            # Target validation
            target_results = validator.validate_target_distribution(split_df)
            logger.info(f"{split_name} target balance: {target_results.get('class_balance_ratio', 'N/A')}")
        
        # Step 4: Model Training with Hyperparameter Tuning
        logger.info("=" * 50)
        logger.info("Step 4: Training model with hyperparameter tuning")
        logger.info("=" * 50)
        
        # Prepare data for training
        train_df = splits['train']
        val_df = splits.get('validation', splits['test'])
        test_df = splits['test']
        
        feature_columns = [col for col in train_df.columns 
                          if col not in ['target', 'data_version', 'generated_timestamp']]
        
        X_train = train_df[feature_columns]
        y_train = train_df['target']
        X_val = val_df[feature_columns]
        y_val = val_df['target']
        X_test = test_df[feature_columns]
        y_test = test_df['target']
        
        # Hyperparameter tuning
        tuner = HyperparameterTuner(config.training)
        logger.info("Starting hyperparameter tuning...")
        
        best_params, best_metrics = tuner.tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            model_type="random_forest",
            max_trials=5  # Reduced for example
        )
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation metrics: {best_metrics}")
        
        # Train final model with best hyperparameters
        config.training.hyperparameters = best_params
        trainer = ModelTrainer(config.training)
        
        model, train_metrics = trainer.train_model(
            X_train, y_train, X_val, y_val,
            model_type="random_forest",
            hyperparameters=best_params
        )
        
        logger.info(f"Training metrics: {train_metrics}")
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(model, X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Step 5: Model Registry
        logger.info("=" * 50)
        logger.info("Step 5: Registering model")
        logger.info("=" * 50)
        
        registry = ModelRegistry(config.training)
        
        # Register the trained model
        run_id = registry.register_model(
            model=model,
            model_name="example-classifier",
            metrics=test_metrics,
            hyperparameters=best_params,
            metadata={
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": len(feature_columns),
                "framework": "scikit-learn"
            },
            tags={
                "experiment": "pipeline-example",
                "dataset": "synthetic"
            }
        )
        
        logger.info(f"Model registered with run_id: {run_id}")
        
        # List registered models
        models = registry.list_models()
        logger.info(f"Total registered models: {len(models)}")
        
        # Step 6: Model Serving
        logger.info("=" * 50)
        logger.info("Step 6: Starting model serving")
        logger.info("=" * 50)
        
        # Start metrics collector
        metrics_collector = MetricsCollector(config.monitoring)
        metrics_collector.start_metrics_server()
        logger.info(f"Metrics server started on port {config.monitoring.metrics_port}")
        
        # Start model server
        with ModelServer(config.serving) as server:
            # Deploy the model
            deployment_name = server.deploy_model(
                model_name="example-classifier",
                model_version="latest",
                num_replicas=1  # Reduced for example
            )
            
            logger.info(f"Model deployed as: {deployment_name}")
            logger.info(f"Model server running at: http://{config.serving.host}:{config.serving.port}")
            
            # Wait for deployment to be ready
            time.sleep(10)
            
            # Step 7: Make sample predictions
            logger.info("=" * 50)
            logger.info("Step 7: Making sample predictions")
            logger.info("=" * 50)
            
            # Get deployment handle for direct prediction
            deployments = server.list_deployments()
            logger.info(f"Active deployments: {list(deployments.keys())}")
            
            # Create sample data for prediction
            sample_data = X_test.head(3).to_dict('records')
            logger.info(f"Sample input: {sample_data[0]}")
            
            # For this example, we'll show how predictions would be made
            # In a real scenario, you'd make HTTP requests to the serving endpoint
            logger.info("Predictions would be made via HTTP API endpoints")
            logger.info(f"Prediction endpoint: http://{config.serving.host}:{config.serving.port}/predict")
            
            # Step 8: Monitor system
            logger.info("=" * 50)
            logger.info("Step 8: Monitoring system")
            logger.info("=" * 50)
            
            # Get deployment metrics
            deployment_metrics = server.get_deployment_metrics(deployment_name)
            logger.info(f"Deployment metrics: {deployment_metrics}")
            
            # Get system metrics
            current_metrics = metrics_collector.get_current_metrics()
            logger.info(f"Collected {len(current_metrics)} metrics")
            
            logger.info(f"Prometheus metrics available at: http://localhost:{config.monitoring.metrics_port}/metrics")
            
            # Let the system run for a bit
            logger.info("System running... (Press Ctrl+C to stop)")
            time.sleep(30)
        
        # Cleanup
        metrics_collector.stop_metrics_server()
        logger.info("Metrics server stopped")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Pipeline completed")


if __name__ == "__main__":
    main()