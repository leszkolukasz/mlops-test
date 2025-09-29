"""Data generation utilities for creating synthetic datasets."""

import numpy as np
import pandas as pd
import ray
from typing import Tuple, Optional
from sklearn.datasets import make_classification, make_regression
from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import DataConfig


@ray.remote
class DataGeneratorActor(LoggerMixin):
    """Ray actor for distributed data generation."""
    
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        
    def generate_classification_batch(
        self, 
        n_samples: int,
        n_features: int = 20,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        n_clusters_per_class: int = 1,
        class_sep: float = 1.0,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of classification data."""
        # Auto-configure feature distribution if not specified
        if n_informative is None:
            n_informative = max(2, min(10, n_features // 2))
        if n_redundant is None:
            n_redundant = max(0, min(5, (n_features - n_informative) // 2))
        
        # Ensure valid feature configuration
        if n_informative + n_redundant > n_features:
            n_informative = max(2, n_features // 2)
            n_redundant = max(0, n_features - n_informative - 1)
        
        self.log_info(
            "Generating classification batch",
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=random_state
        )
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            random_state=random_state or self.config.random_state
        )
        
        return X, y
    
    def generate_regression_batch(
        self,
        n_samples: int,
        n_features: int = 20,
        n_informative: int = 10,
        noise: float = 0.1,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of regression data."""
        self.log_info(
            "Generating regression batch",
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state or self.config.random_state
        )
        
        return X, y


class DataGenerator(LoggerMixin):
    """High-level data generator with Ray-based distributed generation."""
    
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        
    def generate_classification_dataset(
        self,
        n_samples: int = 10000,
        n_features: int = 20,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        n_clusters_per_class: int = 1,
        class_sep: float = 1.0,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Generate a large classification dataset using distributed workers."""
        # Auto-configure feature distribution if not specified
        if n_informative is None:
            n_informative = max(2, min(10, n_features // 2))
        if n_redundant is None:
            n_redundant = max(0, min(5, (n_features - n_informative) // 2))
        
        # Ensure valid feature configuration
        if n_informative + n_redundant > n_features:
            n_informative = max(2, n_features // 2)
            n_redundant = max(0, n_features - n_informative - 1)
        
        self.log_info(
            "Starting distributed classification data generation",
            total_samples=n_samples,
            n_workers=n_workers,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant
        )
        
        # Calculate samples per worker
        samples_per_worker = n_samples // n_workers
        remaining_samples = n_samples % n_workers
        
        # Create Ray actors
        actors = [
            DataGeneratorActor.remote(self.config) 
            for _ in range(n_workers)
        ]
        
        # Generate data in parallel
        futures = []
        for i, actor in enumerate(actors):
            worker_samples = samples_per_worker
            if i < remaining_samples:
                worker_samples += 1
            
            future = actor.generate_classification_batch.remote(
                n_samples=worker_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_clusters_per_class=n_clusters_per_class,
                class_sep=class_sep,
                random_state=self.config.random_state + i
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Combine all batches
        X_combined = np.vstack([X for X, _ in results])
        y_combined = np.hstack([y for _, y in results])
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_combined, columns=feature_names)
        df["target"] = y_combined
        
        # Add metadata columns
        df["data_version"] = "1.0"
        df["generated_timestamp"] = pd.Timestamp.now()
        
        self.log_info(
            "Classification dataset generated successfully",
            total_samples=len(df),
            n_features=n_features,
            target_distribution=df["target"].value_counts().to_dict()
        )
        
        return df
    
    def generate_regression_dataset(
        self,
        n_samples: int = 10000,
        n_features: int = 20,
        n_informative: int = 10,
        noise: float = 0.1,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Generate a large regression dataset using distributed workers."""
        self.log_info(
            "Starting distributed regression data generation",
            total_samples=n_samples,
            n_workers=n_workers
        )
        
        # Calculate samples per worker
        samples_per_worker = n_samples // n_workers
        remaining_samples = n_samples % n_workers
        
        # Create Ray actors
        actors = [
            DataGeneratorActor.remote(self.config) 
            for _ in range(n_workers)
        ]
        
        # Generate data in parallel
        futures = []
        for i, actor in enumerate(actors):
            worker_samples = samples_per_worker
            if i < remaining_samples:
                worker_samples += 1
            
            future = actor.generate_regression_batch.remote(
                n_samples=worker_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise=noise,
                random_state=self.config.random_state + i
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Combine all batches
        X_combined = np.vstack([X for X, _ in results])
        y_combined = np.hstack([y for _, y in results])
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_combined, columns=feature_names)
        df["target"] = y_combined
        
        # Add metadata columns
        df["data_version"] = "1.0"
        df["generated_timestamp"] = pd.Timestamp.now()
        
        self.log_info(
            "Regression dataset generated successfully",
            total_samples=len(df),
            n_features=n_features,
            target_stats={
                "mean": float(df["target"].mean()),
                "std": float(df["target"].std()),
                "min": float(df["target"].min()),
                "max": float(df["target"].max())
            }
        )
        
        return df