"""Data and model drift detection."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.model_selection import train_test_split
import warnings

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import MonitoringConfig


class DriftDetector(LoggerMixin):
    """Detects data drift and model drift."""
    
    def __init__(self, config: MonitoringConfig):
        super().__init__()
        self.config = config
        self.reference_statistics = {}
        self.drift_threshold = config.drift_detection_threshold
        
    def fit_reference(self, reference_data: pd.DataFrame) -> None:
        """Fit the drift detector on reference data."""
        self.log_info("Fitting drift detector on reference data", samples=len(reference_data))
        
        self.reference_statistics = {}
        
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['target', 'prediction']:
                continue
                
            column_data = reference_data[column].dropna()
            
            self.reference_statistics[column] = {
                'mean': float(column_data.mean()),
                'std': float(column_data.std()),
                'min': float(column_data.min()),
                'max': float(column_data.max()),
                'q25': float(column_data.quantile(0.25)),
                'q50': float(column_data.quantile(0.50)),
                'q75': float(column_data.quantile(0.75)),
                'skewness': float(column_data.skew()),
                'kurtosis': float(column_data.kurtosis())
            }
        
        self.log_info(
            "Reference statistics calculated",
            features=len(self.reference_statistics)
        )
    
    def detect_data_drift(
        self, 
        current_data: pd.DataFrame,
        method: str = "ks_test"
    ) -> Dict[str, Any]:
        """Detect data drift between reference and current data."""
        if not self.reference_statistics:
            raise ValueError("Drift detector not fitted. Call fit_reference() first.")
        
        self.log_info(
            "Detecting data drift",
            current_samples=len(current_data),
            method=method
        )
        
        drift_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'method': method,
            'drift_threshold': self.drift_threshold,
            'features_checked': [],
            'drifted_features': [],
            'drift_scores': {},
            'overall_drift_detected': False
        }
        
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in self.reference_statistics:
                continue
                
            if column in ['target', 'prediction']:
                continue
            
            current_column_data = current_data[column].dropna()
            
            if len(current_column_data) == 0:
                continue
            
            drift_results['features_checked'].append(column)
            
            if method == "ks_test":
                drift_score = self._ks_test_drift(column, current_column_data)
            elif method == "statistical":
                drift_score = self._statistical_drift(column, current_column_data)
            elif method == "psi":
                drift_score = self._psi_drift(column, current_column_data)
            else:
                self.log_warning(f"Unknown drift detection method: {method}")
                continue
            
            drift_results['drift_scores'][column] = drift_score
            
            # Check if drift exceeds threshold
            if drift_score.get('drift_detected', False):
                drift_results['drifted_features'].append(column)
        
        drift_results['overall_drift_detected'] = len(drift_results['drifted_features']) > 0
        
        self.log_info(
            "Data drift detection completed",
            features_checked=len(drift_results['features_checked']),
            drifted_features=len(drift_results['drifted_features']),
            overall_drift=drift_results['overall_drift_detected']
        )
        
        return drift_results
    
    def _ks_test_drift(self, column: str, current_data: pd.Series) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        ref_stats = self.reference_statistics[column]
        
        # Generate reference sample from statistics (approximation)
        # In practice, you'd store the actual reference data
        np.random.seed(42)
        ref_sample = np.random.normal(
            ref_stats['mean'], 
            ref_stats['std'], 
            size=len(current_data)
        )
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(ref_sample, current_data)
        
        drift_detected = p_value < 0.05  # 5% significance level
        
        return {
            'method': 'ks_test',
            'ks_statistic': float(ks_statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'drift_score': float(ks_statistic)
        }
    
    def _statistical_drift(self, column: str, current_data: pd.Series) -> Dict[str, Any]:
        """Statistical drift detection based on mean and std changes."""
        ref_stats = self.reference_statistics[column]
        
        current_mean = current_data.mean()
        current_std = current_data.std()
        
        # Normalized differences
        mean_drift = abs(current_mean - ref_stats['mean']) / ref_stats['std']
        std_drift = abs(current_std - ref_stats['std']) / ref_stats['std']
        
        # Combined drift score
        drift_score = (mean_drift + std_drift) / 2
        drift_detected = drift_score > self.drift_threshold
        
        return {
            'method': 'statistical',
            'mean_drift': float(mean_drift),
            'std_drift': float(std_drift),
            'drift_score': float(drift_score),
            'drift_detected': drift_detected,
            'current_mean': float(current_mean),
            'current_std': float(current_std),
            'reference_mean': ref_stats['mean'],
            'reference_std': ref_stats['std']
        }
    
    def _psi_drift(self, column: str, current_data: pd.Series) -> Dict[str, Any]:
        """Population Stability Index (PSI) for drift detection."""
        ref_stats = self.reference_statistics[column]
        
        # Create bins based on reference quantiles
        bins = [
            -np.inf,
            ref_stats['q25'],
            ref_stats['q50'], 
            ref_stats['q75'],
            np.inf
        ]
        
        # Generate reference distribution (approximation)
        np.random.seed(42)
        ref_sample = np.random.normal(
            ref_stats['mean'],
            ref_stats['std'],
            size=10000
        )
        
        # Calculate expected proportions from reference
        ref_counts, _ = np.histogram(ref_sample, bins=bins)
        ref_props = ref_counts / len(ref_sample)
        
        # Calculate actual proportions from current data
        current_counts, _ = np.histogram(current_data, bins=bins)
        current_props = current_counts / len(current_data)
        
        # Calculate PSI
        psi = 0
        for i in range(len(ref_props)):
            if ref_props[i] > 0 and current_props[i] > 0:
                psi += (current_props[i] - ref_props[i]) * np.log(current_props[i] / ref_props[i])
        
        # PSI thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
        drift_detected = psi > 0.2
        
        return {
            'method': 'psi',
            'psi_score': float(psi),
            'drift_detected': drift_detected,
            'drift_score': float(psi),
            'interpretation': self._interpret_psi(psi)
        }
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI score."""
        if psi < 0.1:
            return "no_drift"
        elif psi < 0.2:
            return "moderate_drift"
        else:
            return "significant_drift"
    
    def detect_model_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
        reference_actuals: Optional[pd.Series] = None,
        current_actuals: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Detect model performance drift."""
        self.log_info(
            "Detecting model drift",
            reference_predictions=len(reference_predictions),
            current_predictions=len(current_predictions)
        )
        
        drift_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'prediction_drift': {},
            'performance_drift': {},
            'overall_drift_detected': False
        }
        
        # Prediction distribution drift
        try:
            ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)
            drift_results['prediction_drift'] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': p_value < 0.05,
                'method': 'ks_test'
            }
        except Exception as e:
            self.log_warning("Failed to calculate prediction drift", error=str(e))
        
        # Performance drift (if actuals are available)
        if reference_actuals is not None and current_actuals is not None:
            try:
                drift_results['performance_drift'] = self._calculate_performance_drift(
                    reference_predictions, current_predictions,
                    reference_actuals, current_actuals
                )
            except Exception as e:
                self.log_warning("Failed to calculate performance drift", error=str(e))
        
        # Overall drift assessment
        prediction_drift = drift_results['prediction_drift'].get('drift_detected', False)
        performance_drift = drift_results['performance_drift'].get('drift_detected', False)
        
        drift_results['overall_drift_detected'] = prediction_drift or performance_drift
        
        self.log_info(
            "Model drift detection completed",
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            overall_drift=drift_results['overall_drift_detected']
        )
        
        return drift_results
    
    def _calculate_performance_drift(
        self,
        ref_predictions: pd.Series,
        curr_predictions: pd.Series,
        ref_actuals: pd.Series,
        curr_actuals: pd.Series
    ) -> Dict[str, Any]:
        """Calculate performance drift between reference and current periods."""
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        # Determine if classification or regression
        is_classification = len(pd.concat([ref_actuals, curr_actuals]).unique()) < 20
        
        if is_classification:
            ref_accuracy = accuracy_score(ref_actuals, ref_predictions.round())
            curr_accuracy = accuracy_score(curr_actuals, curr_predictions.round())
            
            accuracy_drift = abs(ref_accuracy - curr_accuracy)
            drift_detected = accuracy_drift > self.config.performance_threshold
            
            return {
                'task_type': 'classification',
                'reference_accuracy': float(ref_accuracy),
                'current_accuracy': float(curr_accuracy),
                'accuracy_drift': float(accuracy_drift),
                'drift_detected': drift_detected,
                'drift_threshold': self.config.performance_threshold
            }
        else:
            ref_mse = mean_squared_error(ref_actuals, ref_predictions)
            curr_mse = mean_squared_error(curr_actuals, curr_predictions)
            ref_r2 = r2_score(ref_actuals, ref_predictions)
            curr_r2 = r2_score(curr_actuals, curr_predictions)
            
            mse_drift = abs(curr_mse - ref_mse) / ref_mse if ref_mse > 0 else 0
            r2_drift = abs(ref_r2 - curr_r2)
            
            # Use R2 drift as primary metric
            drift_detected = r2_drift > self.config.performance_threshold
            
            return {
                'task_type': 'regression',
                'reference_mse': float(ref_mse),
                'current_mse': float(curr_mse),
                'reference_r2': float(ref_r2),
                'current_r2': float(curr_r2),
                'mse_drift': float(mse_drift),
                'r2_drift': float(r2_drift),
                'drift_detected': drift_detected,
                'drift_threshold': self.config.performance_threshold
            }
    
    def generate_drift_report(
        self, 
        drift_results: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate drift detection report."""
        from pathlib import Path
        import json
        
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create detailed report
        report = {
            'summary': {
                'timestamp': drift_results.get('timestamp'),
                'overall_drift_detected': drift_results.get('overall_drift_detected', False),
                'features_checked': len(drift_results.get('features_checked', [])),
                'drifted_features': len(drift_results.get('drifted_features', [])),
                'drift_threshold': drift_results.get('drift_threshold')
            },
            'detailed_results': drift_results
        }
        
        # Save as JSON
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_info(
            "Drift report generated",
            report_path=str(report_path),
            drift_detected=report['summary']['overall_drift_detected']
        )
        
        return str(report_path)