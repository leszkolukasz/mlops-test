"""
MLOps Pipeline using Metaflow.
Orchestrates feature engineering, model training with hyperparameter tuning, and model versioning.
"""
from metaflow import FlowSpec, step, Parameter, card, current
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split


class IrisMLOpsFlow(FlowSpec):
    """
    MLOps pipeline for Iris classification.
    
    Steps:
    1. Load and engineer features
    2. Hyperparameter tuning with Ray Tune
    3. Train final model
    4. Evaluate and log to MLflow
    5. Register model
    """
    
    # Parameters
    test_size = Parameter('test-size', 
                          help='Test set size', 
                          default=0.2)
    
    val_size = Parameter('val-size',
                         help='Validation set size from training data',
                         default=0.2)
    
    n_trials = Parameter('n-trials',
                        help='Number of hyperparameter tuning trials',
                        default=10)
    
    mlflow_tracking_uri = Parameter('mlflow-uri',
                                   help='MLflow tracking URI',
                                   default='file:./mlruns')
    
    @step
    def start(self):
        """
        Initialize the pipeline and set up MLflow.
        """
        print("Starting MLOps Pipeline")
        print(f"Run ID: {current.run_id}")
        print(f"Test size: {self.test_size}")
        print(f"Validation size: {self.val_size}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("iris-classification")
        
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        """
        Load data and perform feature engineering.
        """
        print("Feature Engineering Step")
        
        self.engineer = FeatureEngineer()
        
        # Load and process data
        df = self.engineer.load_data()
        print(f"Loaded {len(df)} samples")
        
        # Create features
        df = self.engineer.create_features(df)
        print(f"Created features. Total features: {len(df.columns) - 1}")
        
        # Split into train and test
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=42, 
            stratify=df['target']
        )
        
        # Further split train into train and validation
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=42,
            stratify=train_df['target']
        )
        
        # Prepare data
        self.X_train, self.y_train = self.engineer.prepare_data(train_df, fit_scaler=True)
        self.X_val, self.y_val = self.engineer.prepare_data(val_df, fit_scaler=False)
        self.X_test, self.y_test = self.engineer.prepare_data(test_df, fit_scaler=False)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Save scaler for deployment
        self.engineer.save_scaler('models/scaler.pkl')
        
        self.next(self.hyperparameter_tuning)
    
    @step
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning using Ray Tune.
        """
        print("Hyperparameter Tuning Step")
        
        # Import Ray Tune
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        import ray
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)
        
        # Define search space
        search_space = {
            'n_estimators': tune.choice([50, 100, 200]),
            'max_depth': tune.randint(5, 20),
            'min_samples_split': tune.randint(2, 10),
            'min_samples_leaf': tune.randint(1, 5)
        }
        
        # Training function for Ray Tune
        def train_fn(config):
            from ray import train as ray_train
            from src.models.train_model import ModelTrainer
            
            trainer = ModelTrainer()
            params = {
                'n_estimators': config['n_estimators'],
                'max_depth': config['max_depth'],
                'min_samples_split': config['min_samples_split'],
                'min_samples_leaf': config['min_samples_leaf'],
                'random_state': 42
            }
            
            trainer.train(self.X_train, self.y_train, params)
            metrics = trainer.evaluate(self.X_val, self.y_val)
            
            ray_train.report({'accuracy': metrics['accuracy']})
        
        # Run hyperparameter search
        tuner = tune.Tuner(
            train_fn,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric='accuracy',
                mode='max',
                num_samples=self.n_trials,
                search_alg=OptunaSearch()
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric='accuracy', mode='max')
        
        self.best_params = best_result.config
        print(f"Best parameters: {self.best_params}")
        print(f"Best validation accuracy: {best_result.metrics['accuracy']:.4f}")
        
        # Shutdown Ray
        ray.shutdown()
        
        self.next(self.train_final_model)
    
    @step
    def train_final_model(self):
        """
        Train the final model with best hyperparameters.
        """
        print("Training Final Model")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"iris-model-{current.run_id}"):
            # Log parameters
            mlflow.log_params(self.best_params)
            mlflow.log_param('test_size', self.test_size)
            mlflow.log_param('val_size', self.val_size)
            
            # Train model
            self.trainer = ModelTrainer()
            params = {
                'n_estimators': self.best_params['n_estimators'],
                'max_depth': self.best_params['max_depth'],
                'min_samples_split': self.best_params['min_samples_split'],
                'min_samples_leaf': self.best_params['min_samples_leaf'],
                'random_state': 42
            }
            
            self.trainer.train(self.X_train, self.y_train, params)
            
            # Evaluate on validation set
            val_metrics = self.trainer.evaluate(self.X_val, self.y_val)
            print("Validation metrics:")
            for key, value in val_metrics.items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value:.4f}")
                    mlflow.log_metric(f'val_{key}', value)
            
            # Evaluate on test set
            test_metrics = self.trainer.evaluate(self.X_test, self.y_test)
            print("Test metrics:")
            for key, value in test_metrics.items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value:.4f}")
                    mlflow.log_metric(f'test_{key}', value)
            
            self.test_metrics = test_metrics
            
            # Save and log model
            model_path = 'models/iris_model.pkl'
            self.trainer.save_model(model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                self.trainer.model,
                "model",
                registered_model_name="iris-classifier"
            )
            
            # Log artifacts
            mlflow.log_artifact('models/scaler.pkl')
            
            self.mlflow_run_id = mlflow.active_run().info.run_id
            print(f"MLflow Run ID: {self.mlflow_run_id}")
        
        self.next(self.end)
    
    @card
    @step
    def end(self):
        """
        End the pipeline and summarize results.
        """
        print("\n" + "="*50)
        print("Pipeline Complete!")
        print("="*50)
        print(f"\nMetaflow Run ID: {current.run_id}")
        print(f"MLflow Run ID: {self.mlflow_run_id}")
        print(f"\nBest Hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nFinal Test Metrics:")
        for key, value in self.test_metrics.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")
        
        print("\nModel saved to: models/iris_model.pkl")
        print("Scaler saved to: models/scaler.pkl")
        print(f"MLflow tracking URI: {self.mlflow_tracking_uri}")


if __name__ == '__main__':
    IrisMLOpsFlow()
