"""Command-line interface for the MLOps pipeline."""

import click
import pandas as pd
import ray
from pathlib import Path
import yaml
import uvicorn

from mlops_pipeline.utils.config import Config
from mlops_pipeline.utils.logger import configure_logging, get_logger
from mlops_pipeline.data.generator import DataGenerator
from mlops_pipeline.data.processor import DataProcessor
from mlops_pipeline.data.validator import DataValidator
from mlops_pipeline.training.trainer import ModelTrainer
from mlops_pipeline.training.hyperparameter_tuner import HyperparameterTuner
from mlops_pipeline.training.model_registry import ModelRegistry
from mlops_pipeline.serving.model_server import ModelServer


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-format', default='json', help='Log format (json or text)')
@click.pass_context
def main(ctx, config, log_level, log_format):
    """MLOps Pipeline CLI."""
    # Configure logging
    configure_logging(log_level=log_level, log_format=log_format)
    
    logger = get_logger("cli")
    logger.info("Starting MLOps Pipeline CLI")
    
    # Load configuration
    if config:
        ctx.obj = Config.from_yaml(config)
    else:
        ctx.obj = Config.from_env()
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray_config = ctx.obj.ray
        ray.init(
            num_cpus=ray_config.num_cpus,
            num_gpus=ray_config.num_gpus,
            dashboard_host=ray_config.dashboard_host,
            dashboard_port=ray_config.dashboard_port,
            ignore_reinit_error=True
        )
        logger.info("Ray initialized", dashboard_port=ray_config.dashboard_port)
    
    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option('--samples', '-n', default=10000, help='Number of samples to generate')
@click.option('--features', '-f', default=20, help='Number of features')
@click.option('--task-type', default='classification', type=click.Choice(['classification', 'regression']))
@click.option('--output', '-o', default='data/raw/dataset.parquet', help='Output file path')
@click.option('--workers', default=4, help='Number of parallel workers')
@click.pass_obj
def generate_data(config, samples, features, task_type, output, workers):
    """Generate synthetic dataset."""
    logger = get_logger("generate_data")
    logger.info(
        "Generating dataset",
        samples=samples,
        features=features,
        task_type=task_type,
        workers=workers
    )
    
    # Create output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    data_generator = DataGenerator(config.data)
    
    if task_type == 'classification':
        df = data_generator.generate_classification_dataset(
            n_samples=samples,
            n_features=features,
            n_workers=workers
        )
    else:
        df = data_generator.generate_regression_dataset(
            n_samples=samples,
            n_features=features,
            n_workers=workers
        )
    
    # Save dataset
    df.to_parquet(output_path, index=False)
    
    logger.info(
        "Dataset generated and saved",
        output_path=str(output_path),
        shape=df.shape
    )
    
    click.echo(f"Dataset saved to {output_path}")
    click.echo(f"Shape: {df.shape}")


@main.command()
@click.option('--input', '-i', default='data/raw/dataset.parquet', help='Input dataset path')
@click.option('--output', '-o', default='data/processed/', help='Output directory')
@click.option('--workers', default=4, help='Number of parallel workers')
@click.pass_obj
def process_data(config, input, output, workers):
    """Process and split dataset."""
    logger = get_logger("process_data")
    logger.info("Processing dataset", input_path=input, output_dir=output)
    
    # Load dataset
    df = pd.read_parquet(input)
    logger.info("Dataset loaded", shape=df.shape)
    
    # Process data
    processor = DataProcessor(config.data)
    processed_df = processor.process_dataset(df, n_workers=workers)
    
    # Create splits
    splits = processor.create_train_test_split(processed_df)
    
    # Save processed data
    saved_files = processor.save_processed_data(splits, output)
    
    logger.info("Data processing completed", saved_files=saved_files)
    
    click.echo(f"Processed data saved to {output}")
    for split_name, split_df in splits.items():
        click.echo(f"  {split_name}: {len(split_df)} samples")


@main.command()
@click.option('--data-dir', '-d', default='data/processed/', help='Processed data directory')
@click.option('--output', '-o', default='reports/validation_report.html', help='Output report path')
@click.pass_obj
def validate_data(config, data_dir, output):
    """Validate data quality."""
    logger = get_logger("validate_data")
    logger.info("Validating data", data_dir=data_dir)
    
    # Load processed data
    processor = DataProcessor(config.data)
    splits = processor.load_processed_data(data_dir)
    
    validator = DataValidator(config.data)
    
    validation_results = {}
    
    for split_name, split_df in splits.items():
        logger.info(f"Validating {split_name} split", samples=len(split_df))
        
        # Schema validation
        expected_columns = [col for col in split_df.columns if col != 'target']
        schema_validation = validator.validate_schema(split_df, expected_columns)
        
        # Quality validation
        quality_validation = validator.validate_data_quality(split_df)
        
        # Target validation
        target_validation = validator.validate_target_distribution(split_df)
        
        validation_results[split_name] = {
            'schema': schema_validation,
            'quality': quality_validation,
            'target': target_validation
        }
    
    # Generate report
    report_path = validator.generate_validation_report(validation_results, output)
    
    logger.info("Data validation completed", report_path=report_path)
    click.echo(f"Validation report saved to {report_path}")


@main.command()
@click.option('--data-dir', '-d', default='data/processed/', help='Processed data directory')
@click.option('--model-type', default='random_forest', help='Model type to train')
@click.option('--tune', is_flag=True, help='Perform hyperparameter tuning')
@click.option('--max-trials', default=10, help='Maximum tuning trials')
@click.pass_obj
def train_model(config, data_dir, model_type, tune, max_trials):
    """Train a machine learning model."""
    logger = get_logger("train_model")
    logger.info(
        "Starting model training",
        model_type=model_type,
        tune_hyperparameters=tune,
        max_trials=max_trials
    )
    
    # Load processed data
    processor = DataProcessor(config.data)
    splits = processor.load_processed_data(data_dir)
    
    train_df = splits['train']
    val_df = splits.get('validation')
    test_df = splits.get('test')
    
    # Prepare features and targets
    feature_columns = [col for col in train_df.columns 
                      if col not in ['target', 'data_version', 'generated_timestamp']]
    
    X_train = train_df[feature_columns]
    y_train = train_df['target']
    
    X_val = val_df[feature_columns] if val_df is not None else None
    y_val = val_df['target'] if val_df is not None else None
    
    X_test = test_df[feature_columns] if test_df is not None else None
    y_test = test_df['target'] if test_df is not None else None
    
    # Update training config
    config.training.model_type = model_type
    
    # Hyperparameter tuning
    if tune:
        logger.info("Starting hyperparameter tuning")
        tuner = HyperparameterTuner(config.training)
        tuner.config.max_trials = max_trials
        
        if X_val is not None:
            best_params, best_metrics = tuner.tune_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type=model_type,
                max_trials=max_trials
            )
        else:
            best_params, best_metrics = tuner.tune_with_cross_validation(
                X_train, y_train,
                model_type=model_type,
                max_trials=max_trials
            )
        
        config.training.hyperparameters = best_params
        logger.info("Hyperparameter tuning completed", best_params=best_params)
    
    # Train model
    trainer = ModelTrainer(config.training)
    model, train_metrics = trainer.train_model(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        hyperparameters=config.training.hyperparameters
    )
    
    # Evaluate on test set if available
    if X_test is not None:
        test_metrics = trainer.evaluate_model(model, X_test, y_test)
        logger.info("Test evaluation completed", test_metrics=test_metrics)
        click.echo("Test Metrics:")
        for metric, value in test_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
    
    logger.info("Model training completed", train_metrics=train_metrics)
    click.echo("Training completed successfully!")


@main.command()
@click.option('--model-name', default='mlops-model', help='Model name to serve')
@click.option('--model-version', default='latest', help='Model version to serve')
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--replicas', default=2, help='Number of replicas')
@click.pass_obj
def serve_model(config, model_name, model_version, host, port, replicas):
    """Start model serving server."""
    logger = get_logger("serve_model")
    logger.info(
        "Starting model serving",
        model_name=model_name,
        model_version=model_version,
        host=host,
        port=port,
        replicas=replicas
    )
    
    # Update serving config
    config.serving.host = host
    config.serving.port = port
    config.serving.num_replicas = replicas
    config.serving.model_name = model_name
    config.serving.model_version = model_version
    
    # Start model server
    with ModelServer(config.serving) as server:
        # Deploy the model
        deployment_name = server.deploy_model(
            model_name=model_name,
            model_version=model_version,
            num_replicas=replicas
        )
        
        logger.info("Model deployed", deployment_name=deployment_name)
        click.echo(f"Model server started at http://{host}:{port}")
        click.echo(f"Model {model_name}:{model_version} deployed as {deployment_name}")
        
        # Keep server running
        try:
            import signal
            import sys
            
            def signal_handler(sig, frame):
                logger.info("Shutting down model server")
                click.echo("\nShutting down model server...")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Wait indefinitely
            signal.pause()
            
        except (KeyboardInterrupt, SystemExit):
            logger.info("Model server shutdown requested")
            click.echo("Model server stopped.")


@main.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8080, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_obj
def start_api(config, host, port, reload):
    """Start the FastAPI server."""
    logger = get_logger("start_api")
    logger.info("Starting FastAPI server", host=host, port=port)
    
    click.echo(f"Starting MLOps API server at http://{host}:{port}")
    click.echo(f"API documentation available at http://{host}:{port}/docs")
    
    # Start FastAPI server
    uvicorn.run(
        "mlops_pipeline.serving.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@main.command()
@click.option('--input', '-i', default='config/default.yaml', help='Input config file')
@click.option('--output', '-o', help='Output config file (default: same as input)')
@click.pass_obj
def generate_config(config, input, output):
    """Generate configuration file."""
    output = output or input
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config.save_yaml(output)
    
    click.echo(f"Configuration saved to {output}")


@main.command()
@click.pass_obj
def ray_dashboard(config):
    """Show Ray dashboard URL."""
    if ray.is_initialized():
        dashboard_url = f"http://{config.ray.dashboard_host}:{config.ray.dashboard_port}"
        click.echo(f"Ray Dashboard: {dashboard_url}")
    else:
        click.echo("Ray is not initialized")


@main.command()
@click.pass_obj
def status(config):
    """Show system status."""
    click.echo("MLOps Pipeline Status")
    click.echo("=" * 50)
    
    # Ray status
    if ray.is_initialized():
        cluster_resources = ray.cluster_resources()
        click.echo(f"Ray Status: ✓ Initialized")
        click.echo(f"  CPUs: {cluster_resources.get('CPU', 0)}")
        click.echo(f"  Memory: {cluster_resources.get('memory', 0) / (1024**3):.1f} GB")
        click.echo(f"  Dashboard: http://{config.ray.dashboard_host}:{config.ray.dashboard_port}")
    else:
        click.echo("Ray Status: ✗ Not initialized")
    
    # Check if model registry is accessible
    try:
        registry = ModelRegistry(config.training)
        models = registry.list_models()
        click.echo(f"Model Registry: ✓ Accessible ({len(models)} models)")
    except Exception as e:
        click.echo(f"Model Registry: ✗ Error - {str(e)}")
    
    # Check data directories
    data_dirs = ['data/raw', 'data/processed', 'models', 'reports']
    for data_dir in data_dirs:
        path = Path(data_dir)
        if path.exists():
            file_count = len(list(path.glob('*')))
            click.echo(f"{data_dir}: ✓ ({file_count} files)")
        else:
            click.echo(f"{data_dir}: ✗ Not found")


if __name__ == '__main__':
    main()