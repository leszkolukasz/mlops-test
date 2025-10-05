"""
Feature engineering module for ML pipeline.
Handles data loading, preprocessing, and feature creation.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class FeatureEngineer:
    """
    Feature engineering class for iris dataset.
    Creates additional features and handles preprocessing.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the iris dataset"""
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        return df
    
    def create_features(self, df):
        """
        Create additional features from raw data.
        
        Features:
        - Petal area (petal length * petal width)
        - Sepal area (sepal length * sepal width)
        - Petal/Sepal ratio
        """
        # Create interaction features
        df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
        df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
        
        # Create ratio features
        df['petal_sepal_length_ratio'] = df['petal length (cm)'] / (df['sepal length (cm)'] + 1e-8)
        df['petal_sepal_width_ratio'] = df['petal width (cm)'] / (df['sepal width (cm)'] + 1e-8)
        
        return df
    
    def prepare_data(self, df, fit_scaler=True):
        """
        Prepare data for modeling.
        
        Args:
            df: DataFrame with features and target
            fit_scaler: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            X_scaled: Scaled features
            y: Target variable
        """
        X = df.drop('target', axis=1)
        y = df['target']
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y
    
    def save_scaler(self, path):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path):
        """Load a fitted scaler"""
        self.scaler = joblib.load(path)
        
    def process_pipeline(self, test_size=0.2, random_state=42):
        """
        Full feature engineering pipeline.
        
        Returns:
            X_train, X_test, y_train, y_test: Split and processed data
        """
        # Load data
        df = self.load_data()
        
        # Create features
        df = self.create_features(df)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['target'])
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_df, fit_scaler=True)
        X_test, y_test = self.prepare_data(test_df, fit_scaler=False)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the feature engineering
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.process_pipeline()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
