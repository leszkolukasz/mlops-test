"""Data processing and feature engineering with Ray Data."""

import pandas as pd
import numpy as np
import ray
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import DataConfig


@ray.remote
def process_batch(
    batch: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    feature_columns: Optional[List[str]] = None,
    fit_scaler: bool = False
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Process a batch of data with scaling and feature engineering."""
    if feature_columns is None:
        feature_columns = [col for col in batch.columns if col not in ["target", "data_version", "generated_timestamp"]]
    
    # Feature engineering
    # Add polynomial features (degree 2) for first few features to demonstrate
    for i in range(min(3, len(feature_columns))):
        col = feature_columns[i]
        batch[f"{col}_squared"] = batch[col] ** 2
    
    # Add interaction features
    if len(feature_columns) >= 2:
        batch["feature_0_x_feature_1"] = batch[feature_columns[0]] * batch[feature_columns[1]]
    
    # Update feature columns list
    numeric_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns_updated = [col for col in numeric_cols if col not in ["target", "data_version"]]
    
    # Scaling
    if fit_scaler:
        scaler = StandardScaler()
        batch[feature_columns_updated] = scaler.fit_transform(batch[feature_columns_updated])
    elif scaler is not None:
        batch[feature_columns_updated] = scaler.transform(batch[feature_columns_updated])
    
    return batch, scaler


class DataProcessor(LoggerMixin):
    """Data processor with Ray-based distributed processing."""
    
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[List[str]] = None
        
    def process_dataset(
        self,
        df: pd.DataFrame,
        fit_transformers: bool = True,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process dataset using distributed Ray workers."""
        self.log_info(
            "Starting distributed data processing",
            input_shape=df.shape,
            n_workers=n_workers,
            fit_transformers=fit_transformers
        )
        
        # Identify feature columns
        self.feature_columns = [
            col for col in df.columns 
            if col not in ["target", "data_version", "generated_timestamp"]
        ]
        
        # Split data into chunks for parallel processing
        chunk_size = len(df) // n_workers
        if chunk_size == 0:
            chunk_size = len(df)
        
        chunks = [
            df.iloc[i:i + chunk_size].copy() 
            for i in range(0, len(df), chunk_size)
        ]
        
        # Process first chunk to fit transformers
        if fit_transformers and chunks:
            first_chunk, self.scaler = ray.get(
                process_batch.remote(
                    chunks[0],
                    scaler=None,
                    feature_columns=self.feature_columns,
                    fit_scaler=True
                )
            )
            processed_chunks = [first_chunk]
            remaining_chunks = chunks[1:]
        else:
            remaining_chunks = chunks
            processed_chunks = []
        
        # Process remaining chunks in parallel
        if remaining_chunks:
            futures = [
                process_batch.remote(
                    chunk,
                    scaler=self.scaler,
                    feature_columns=self.feature_columns,
                    fit_scaler=False
                )
                for chunk in remaining_chunks
            ]
            
            results = ray.get(futures)
            processed_chunks.extend([result[0] for result in results])
        
        # Combine all processed chunks
        processed_df = pd.concat(processed_chunks, ignore_index=True)
        
        self.log_info(
            "Data processing completed",
            output_shape=processed_df.shape,
            feature_columns=len(self.feature_columns),
            scaler_fitted=self.scaler is not None
        )
        
        return processed_df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """Create train/test/validation splits."""
        test_size = test_size or (1 - self.config.train_test_split)
        validation_size = validation_size or self.config.validation_split
        
        self.log_info(
            "Creating train/test/validation splits",
            test_size=test_size,
            validation_size=validation_size,
            total_samples=len(df)
        )
        
        # First split: train + validation vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=df["target"] if df["target"].dtype in ["object", "category"] or df["target"].nunique() < 10 else None
        )
        
        # Second split: train vs validation
        if validation_size > 0:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=validation_size,
                random_state=self.config.random_state,
                stratify=train_val_df["target"] if train_val_df["target"].dtype in ["object", "category"] or train_val_df["target"].nunique() < 10 else None
            )
            
            splits = {
                "train": train_df,
                "validation": val_df,
                "test": test_df
            }
        else:
            splits = {
                "train": train_val_df,
                "test": test_df
            }
        
        # Log split information
        for split_name, split_df in splits.items():
            self.log_info(
                f"{split_name} split created",
                samples=len(split_df),
                target_distribution=split_df["target"].value_counts().to_dict() if split_df["target"].nunique() < 20 else "continuous"
            )
        
        return splits
    
    def save_processed_data(
        self,
        splits: Dict[str, pd.DataFrame],
        output_dir: str
    ) -> Dict[str, str]:
        """Save processed data splits to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for split_name, split_df in splits.items():
            file_path = output_path / f"{split_name}.parquet"
            split_df.to_parquet(file_path, index=False)
            saved_files[split_name] = str(file_path)
            
            self.log_info(
                f"Saved {split_name} split",
                file_path=str(file_path),
                samples=len(split_df)
            )
        
        # Save scaler if available
        if self.scaler is not None:
            scaler_path = output_path / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            saved_files["scaler"] = str(scaler_path)
            
            self.log_info("Saved data scaler", file_path=str(scaler_path))
        
        # Save feature columns
        if self.feature_columns is not None:
            features_path = output_path / "feature_columns.pkl"
            with open(features_path, "wb") as f:
                pickle.dump(self.feature_columns, f)
            saved_files["features"] = str(features_path)
            
            self.log_info("Saved feature columns", file_path=str(features_path))
        
        return saved_files
    
    def load_processed_data(
        self,
        input_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """Load processed data splits from files."""
        input_path = Path(input_dir)
        
        splits = {}
        for file_path in input_path.glob("*.parquet"):
            split_name = file_path.stem
            if split_name in ["train", "validation", "test"]:
                splits[split_name] = pd.read_parquet(file_path)
                
                self.log_info(
                    f"Loaded {split_name} split",
                    file_path=str(file_path),
                    samples=len(splits[split_name])
                )
        
        # Load scaler if available
        scaler_path = input_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.log_info("Loaded data scaler", file_path=str(scaler_path))
        
        # Load feature columns if available
        features_path = input_path / "feature_columns.pkl"
        if features_path.exists():
            with open(features_path, "rb") as f:
                self.feature_columns = pickle.load(f)
            self.log_info("Loaded feature columns", file_path=str(features_path))
        
        return splits
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics for monitoring."""
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.columns 
                if col not in ["target", "data_version", "generated_timestamp"]
            ]
        
        stats = {}
        for col in self.feature_columns:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "null_count": int(df[col].isnull().sum()),
                    "null_percentage": float(df[col].isnull().mean() * 100)
                }
        
        return stats