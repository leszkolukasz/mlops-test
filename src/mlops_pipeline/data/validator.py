"""Data validation and quality checks using Great Expectations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import DataConfig


class DataValidator(LoggerMixin):
    """Data validator for quality checks and validation."""
    
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """Validate DataFrame schema against expected columns."""
        self.log_info("Validating data schema", expected_columns=len(expected_columns))
        
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set
        
        validation_result = {
            "schema_valid": len(missing_columns) == 0,
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "column_count": len(df.columns),
            "expected_column_count": len(expected_columns)
        }
        
        if not validation_result["schema_valid"]:
            self.log_error(
                "Schema validation failed",
                missing_columns=list(missing_columns),
                extra_columns=list(extra_columns)
            )
        else:
            self.log_info("Schema validation passed")
        
        return validation_result
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:
        """Validate DataFrame data types."""
        self.log_info("Validating data types", expected_types=len(expected_types))
        
        type_mismatches = {}
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if expected_type not in actual_type:
                    type_mismatches[column] = {
                        "expected": expected_type,
                        "actual": actual_type
                    }
        
        validation_result = {
            "types_valid": len(type_mismatches) == 0,
            "type_mismatches": type_mismatches,
            "validated_columns": len(expected_types)
        }
        
        if not validation_result["types_valid"]:
            self.log_error("Data type validation failed", type_mismatches=type_mismatches)
        else:
            self.log_info("Data type validation passed")
        
        return validation_result
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks."""
        self.log_info("Starting data quality validation", rows=len(df), columns=len(df.columns))
        
        quality_results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().mean() * 100),
            "column_stats": {}
        }
        
        # Check each column
        for column in df.columns:
            col_stats = {
                "null_count": int(df[column].isnull().sum()),
                "null_percentage": float(df[column].isnull().mean() * 100),
                "unique_count": int(df[column].nunique()),
                "data_type": str(df[column].dtype)
            }
            
            # Numeric column specific checks
            if pd.api.types.is_numeric_dtype(df[column]):
                col_stats.update({
                    "mean": float(df[column].mean()) if not df[column].isnull().all() else None,
                    "std": float(df[column].std()) if not df[column].isnull().all() else None,
                    "min": float(df[column].min()) if not df[column].isnull().all() else None,
                    "max": float(df[column].max()) if not df[column].isnull().all() else None,
                    "zeros_count": int((df[column] == 0).sum()),
                    "negative_count": int((df[column] < 0).sum()),
                    "infinite_count": int(np.isinf(df[column]).sum())
                })
                
                # Check for outliers (IQR method)
                if not df[column].isnull().all():
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                    col_stats["outliers_count"] = int(outliers)
                    col_stats["outliers_percentage"] = float(outliers / len(df) * 100)
            
            quality_results["column_stats"][column] = col_stats
        
        # Overall quality score (simple heuristic)
        quality_score = self._calculate_quality_score(quality_results)
        quality_results["quality_score"] = quality_score
        
        self.log_info(
            "Data quality validation completed",
            quality_score=quality_score,
            duplicate_rows=quality_results["duplicate_rows"],
            columns_with_nulls=sum(1 for stats in quality_results["column_stats"].values() if stats["null_count"] > 0)
        )
        
        return quality_results
    
    def validate_target_distribution(
        self, 
        df: pd.DataFrame, 
        target_column: str = "target"
    ) -> Dict[str, Any]:
        """Validate target variable distribution."""
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}
        
        self.log_info("Validating target distribution", target_column=target_column)
        
        target_series = df[target_column]
        is_classification = target_series.nunique() < 20 and target_series.dtype in ['object', 'category', 'int64']
        
        validation_result = {
            "target_column": target_column,
            "is_classification": is_classification,
            "unique_values": int(target_series.nunique()),
            "null_count": int(target_series.isnull().sum()),
            "null_percentage": float(target_series.isnull().mean() * 100)
        }
        
        if is_classification:
            # Classification target validation
            value_counts = target_series.value_counts()
            validation_result.update({
                "class_distribution": value_counts.to_dict(),
                "class_balance_ratio": float(value_counts.min() / value_counts.max()),
                "majority_class_percentage": float(value_counts.max() / len(df) * 100)
            })
            
            # Check for class imbalance
            imbalance_threshold = 0.1  # Minority class should be at least 10%
            validation_result["is_balanced"] = validation_result["class_balance_ratio"] >= imbalance_threshold
            
        else:
            # Regression target validation
            validation_result.update({
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "skewness": float(target_series.skew()),
                "kurtosis": float(target_series.kurtosis())
            })
            
            # Check for extreme skewness
            validation_result["is_normally_distributed"] = abs(validation_result["skewness"]) < 2
        
        self.log_info(
            "Target distribution validation completed",
            is_classification=is_classification,
            unique_values=validation_result["unique_values"],
            null_percentage=validation_result["null_percentage"]
        )
        
        return validation_result
    
    def validate_feature_drift(
        self, 
        reference_df: pd.DataFrame, 
        current_df: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect feature drift between reference and current datasets."""
        self.log_info(
            "Validating feature drift",
            reference_samples=len(reference_df),
            current_samples=len(current_df),
            threshold=threshold
        )
        
        common_columns = set(reference_df.columns) & set(current_df.columns)
        numeric_columns = [
            col for col in common_columns 
            if pd.api.types.is_numeric_dtype(reference_df[col]) and 
               pd.api.types.is_numeric_dtype(current_df[col])
        ]
        
        drift_results = {
            "total_features_checked": len(numeric_columns),
            "drift_threshold": threshold,
            "features_with_drift": [],
            "feature_drift_scores": {},
            "overall_drift_detected": False
        }
        
        for column in numeric_columns:
            # Calculate statistical distance (normalized difference in means and stds)
            ref_mean = reference_df[column].mean()
            ref_std = reference_df[column].std()
            curr_mean = current_df[column].mean()
            curr_std = current_df[column].std()
            
            # Avoid division by zero
            if ref_std == 0 or curr_std == 0:
                continue
            
            mean_drift = abs(ref_mean - curr_mean) / ref_std
            std_drift = abs(ref_std - curr_std) / ref_std
            
            # Combined drift score
            drift_score = (mean_drift + std_drift) / 2
            
            drift_results["feature_drift_scores"][column] = {
                "drift_score": float(drift_score),
                "mean_drift": float(mean_drift),
                "std_drift": float(std_drift),
                "has_drift": drift_score > threshold,
                "reference_mean": float(ref_mean),
                "reference_std": float(ref_std),
                "current_mean": float(curr_mean),
                "current_std": float(curr_std)
            }
            
            if drift_score > threshold:
                drift_results["features_with_drift"].append(column)
        
        drift_results["overall_drift_detected"] = len(drift_results["features_with_drift"]) > 0
        
        self.log_info(
            "Feature drift validation completed",
            features_with_drift=len(drift_results["features_with_drift"]),
            overall_drift_detected=drift_results["overall_drift_detected"]
        )
        
        return drift_results
    
    def _calculate_quality_score(self, quality_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0
        
        # Penalize for duplicates
        if quality_results["duplicate_percentage"] > 0:
            score -= min(quality_results["duplicate_percentage"] / 100 * 0.2, 0.2)
        
        # Penalize for missing values
        total_null_percentage = np.mean([
            stats["null_percentage"] for stats in quality_results["column_stats"].values()
        ])
        score -= min(total_null_percentage / 100 * 0.3, 0.3)
        
        # Penalize for columns with too many nulls
        high_null_columns = sum(
            1 for stats in quality_results["column_stats"].values() 
            if stats["null_percentage"] > 50
        )
        if high_null_columns > 0:
            score -= min(high_null_columns / len(quality_results["column_stats"]) * 0.2, 0.2)
        
        return max(score, 0.0)
    
    def generate_validation_report(
        self, 
        validation_results: Dict[str, Any], 
        output_path: str
    ) -> str:
        """Generate a comprehensive validation report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HTML report
        html_content = self._generate_html_report(validation_results)
        
        with open(report_path, "w") as f:
            f.write(html_content)
        
        # Also save as JSON
        json_path = report_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        self.log_info(
            "Validation report generated",
            html_report=str(report_path),
            json_report=str(json_path)
        )
        
        return str(report_path)
    
    def _generate_html_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate HTML validation report."""
        # Simple HTML template - in production, you'd use a proper templating engine
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Validation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Quality Score: {validation_results.get('quality_score', 'N/A')}</p>
                <p>Total Rows: {validation_results.get('total_rows', 'N/A')}</p>
                <p>Total Columns: {validation_results.get('total_columns', 'N/A')}</p>
            </div>
        </body>
        </html>
        """
        return html