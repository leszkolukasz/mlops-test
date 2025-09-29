#!/usr/bin/env python3
"""
Quick Start Example for MLOps Pipeline

This script demonstrates the basic workflow:
1. Generate synthetic data
2. Process and validate data
3. Train a model
4. Register the model

Usage:
    python examples/quick_start.py
"""

import ray
from pathlib import Path

from mlops_pipeline.utils.config import Config
from mlops_pipeline.utils.logger import configure_logging, get_logger
from mlops_pipeline.data.generator import DataGenerator
from mlops_pipeline.data.processor import DataProcessor
from mlops_pipeline.training.trainer import ModelTrainer
from mlops_pipeline.training.model_registry import ModelRegistry


def main():
    """Run a quick start example."""
    # Setup
    configure_logging(log_level="INFO", log_format="text")
    logger = get_logger("quick_start")
    
    logger.info("🚀 Starting MLOps Pipeline Quick Start")
    
    # Load configuration
    config = Config()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=2, dashboard_port=8265, ignore_reinit_error=True)
        logger.info("Ray initialized - Dashboard: http://localhost:8265")
    
    try:
        # Step 1: Generate Data
        logger.info("📊 Step 1: Generating synthetic dataset")
        data_generator = DataGenerator(config.data)
        df = data_generator.generate_classification_dataset(
            n_samples=1000,
            n_features=10,
            n_workers=1
        )
        logger.info(f"Generated dataset with shape: {df.shape}")
        
        # Step 2: Process Data
        logger.info("⚙️  Step 2: Processing dataset")
        processor = DataProcessor(config.data)
        processed_df = processor.process_dataset(df, n_workers=1)
        splits = processor.create_train_test_split(processed_df)
        
        logger.info(f"Created splits: {list(splits.keys())}")
        for split_name, split_df in splits.items():
            logger.info(f"  {split_name}: {len(split_df)} samples")
        
        # Step 3: Train Model
        logger.info("🤖 Step 3: Training model")
        train_df = splits['train']
        test_df = splits['test']
        
        feature_columns = [col for col in train_df.columns 
                          if col not in ['target', 'data_version', 'generated_timestamp']]
        
        X_train = train_df[feature_columns]
        y_train = train_df['target']
        X_test = test_df[feature_columns]
        y_test = test_df['target']
        
        trainer = ModelTrainer(config.training)
        model, train_metrics = trainer.train_model(
            X_train, y_train,
            model_type="random_forest"
        )
        
        logger.info(f"Training completed with metrics: {train_metrics}")
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(model, X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Step 4: Register Model
        logger.info("📝 Step 4: Registering model")
        registry = ModelRegistry(config.training)
        
        run_id = registry.register_model(
            model=model,
            model_name="quickstart-classifier",
            metrics=test_metrics,
            metadata={
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": len(feature_columns)
            },
            tags={"experiment": "quickstart"}
        )
        
        logger.info(f"Model registered with run_id: {run_id}")
        
        # Step 5: Show Results
        logger.info("✅ Quick Start Completed Successfully!")
        logger.info(f"📈 Test Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.3f}")
        logger.info(f"📈 Test F1 Score: {test_metrics.get('test_f1', 'N/A'):.3f}")
        logger.info("🌐 Access Ray Dashboard: http://localhost:8265")
        logger.info("🔬 Access MLflow UI: Run 'mlflow ui' and visit http://localhost:5000")
        
        # List registered models
        models = registry.list_models()
        logger.info(f"Total registered models: {len(models)}")
        
        logger.info("🎉 All done! You've successfully run the MLOps pipeline.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Pipeline completed")


if __name__ == "__main__":
    main()