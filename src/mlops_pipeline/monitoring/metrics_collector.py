"""Metrics collection with Prometheus for model monitoring."""

import time
import threading
from typing import Dict, Any, Optional, List
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    start_http_server, generate_latest
)
import psutil
import ray

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import MonitoringConfig


class MetricsCollector(LoggerMixin):
    """Prometheus metrics collector for MLOps pipeline."""
    
    def __init__(self, config: MonitoringConfig):
        super().__init__()
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics_server = None
        self.metrics_thread = None
        self.stop_event = threading.Event()
        
        # Initialize metrics
        self._init_metrics()
        
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Prediction metrics
        self.prediction_counter = Counter(
            'mlops_predictions_total',
            'Total number of predictions made',
            ['model_name', 'model_version', 'status'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'mlops_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.prediction_batch_size = Histogram(
            'mlops_prediction_batch_size',
            'Number of samples in prediction batch',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_deployments = Gauge(
            'mlops_model_deployments',
            'Number of active model deployments',
            registry=self.registry
        )
        
        self.model_replicas = Gauge(
            'mlops_model_replicas',
            'Number of replicas per model deployment',
            ['deployment_name', 'model_name', 'model_version'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'mlops_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'mlops_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'mlops_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'mlops_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Ray cluster metrics
        self.ray_nodes = Gauge(
            'mlops_ray_nodes',
            'Number of Ray nodes',
            registry=self.registry
        )
        
        self.ray_cpus_total = Gauge(
            'mlops_ray_cpus_total',
            'Total Ray CPUs',
            registry=self.registry
        )
        
        self.ray_cpus_used = Gauge(
            'mlops_ray_cpus_used',
            'Used Ray CPUs',
            registry=self.registry
        )
        
        self.ray_memory_total = Gauge(
            'mlops_ray_memory_total_bytes',
            'Total Ray memory in bytes',
            registry=self.registry
        )
        
        self.ray_memory_used = Gauge(
            'mlops_ray_memory_used_bytes',
            'Used Ray memory in bytes',
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'mlops_data_quality_score',
            'Data quality score (0-1)',
            ['dataset_name'],
            registry=self.registry
        )
        
        self.drift_detected = Counter(
            'mlops_drift_detected_total',
            'Number of times drift was detected',
            ['feature_name', 'drift_type'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'mlops_model_accuracy',
            'Model accuracy',
            ['model_name', 'model_version', 'dataset'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'mlops_model_f1_score',
            'Model F1 score',
            ['model_name', 'model_version', 'dataset'],
            registry=self.registry
        )
        
        self.model_mse = Gauge(
            'mlops_model_mse',
            'Model Mean Squared Error',
            ['model_name', 'model_version', 'dataset'],
            registry=self.registry
        )
        
        # Training metrics
        self.training_runs = Counter(
            'mlops_training_runs_total',
            'Total number of training runs',
            ['model_type', 'status'],
            registry=self.registry
        )
        
        self.training_duration = Histogram(
            'mlops_training_duration_seconds',
            'Training duration in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        self.hyperparameter_trials = Counter(
            'mlops_hyperparameter_trials_total',
            'Total number of hyperparameter optimization trials',
            ['model_type'],
            registry=self.registry
        )
        
        # Info metrics
        self.build_info = Info(
            'mlops_build_info',
            'Build information',
            registry=self.registry
        )
        
        self.build_info.info({
            'version': '0.1.0',
            'python_version': '3.9+',
            'ray_version': ray.__version__
        })
        
        self.log_info("Prometheus metrics initialized")
    
    def start_metrics_server(self) -> None:
        """Start Prometheus metrics server."""
        if self.metrics_server is not None:
            self.log_warning("Metrics server already running")
            return
        
        self.log_info(
            "Starting Prometheus metrics server",
            port=self.config.metrics_port
        )
        
        try:
            self.metrics_server = start_http_server(
                self.config.metrics_port,
                registry=self.registry
            )
            
            # Start background metrics collection
            self.metrics_thread = threading.Thread(
                target=self._collect_system_metrics,
                daemon=True
            )
            self.metrics_thread.start()
            
            self.log_info(
                "Prometheus metrics server started",
                port=self.config.metrics_port
            )
            
        except Exception as e:
            self.log_error(
                "Failed to start metrics server",
                error=str(e),
                port=self.config.metrics_port
            )
            raise
    
    def stop_metrics_server(self) -> None:
        """Stop Prometheus metrics server."""
        self.log_info("Stopping Prometheus metrics server")
        
        self.stop_event.set()
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        
        if self.metrics_server:
            self.metrics_server.shutdown()
            self.metrics_server = None
        
        self.log_info("Prometheus metrics server stopped")
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics in background."""
        while not self.stop_event.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                self.memory_usage_percent.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.percent)
                
                # Ray cluster metrics (if Ray is initialized)
                if ray.is_initialized():
                    try:
                        cluster_resources = ray.cluster_resources()
                        cluster_state = ray.state.cluster_resources()
                        
                        # Node count
                        nodes = ray.nodes()
                        self.ray_nodes.set(len([n for n in nodes if n['Alive']]))
                        
                        # CPU metrics
                        total_cpus = cluster_resources.get('CPU', 0)
                        used_cpus = total_cpus - cluster_state.get('CPU', 0)
                        self.ray_cpus_total.set(total_cpus)
                        self.ray_cpus_used.set(used_cpus)
                        
                        # Memory metrics
                        total_memory = cluster_resources.get('memory', 0)
                        used_memory = total_memory - cluster_state.get('memory', 0)
                        self.ray_memory_total.set(total_memory)
                        self.ray_memory_used.set(used_memory)
                        
                    except Exception as e:
                        self.log_debug("Failed to collect Ray metrics", error=str(e))
                
                # Sleep for 10 seconds before next collection
                self.stop_event.wait(10)
                
            except Exception as e:
                self.log_error("Error collecting system metrics", error=str(e))
                self.stop_event.wait(10)
    
    def record_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_time_ms: float,
        num_samples: int,
        success: bool = True
    ) -> None:
        """Record prediction metrics."""
        status = "success" if success else "error"
        
        self.prediction_counter.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        if success:
            self.prediction_duration.labels(
                model_name=model_name,
                model_version=model_version
            ).observe(prediction_time_ms / 1000.0)
            
            self.prediction_batch_size.labels(
                model_name=model_name,
                model_version=model_version
            ).observe(num_samples)
    
    def record_training_run(
        self,
        model_type: str,
        duration_seconds: float,
        success: bool = True
    ) -> None:
        """Record training run metrics."""
        status = "success" if success else "error"
        
        self.training_runs.labels(
            model_type=model_type,
            status=status
        ).inc()
        
        if success:
            self.training_duration.labels(
                model_type=model_type
            ).observe(duration_seconds)
    
    def record_hyperparameter_trial(self, model_type: str) -> None:
        """Record hyperparameter optimization trial."""
        self.hyperparameter_trials.labels(
            model_type=model_type
        ).inc()
    
    def update_model_metrics(
        self,
        model_name: str,
        model_version: str,
        dataset: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update model performance metrics."""
        if 'accuracy' in metrics:
            self.model_accuracy.labels(
                model_name=model_name,
                model_version=model_version,
                dataset=dataset
            ).set(metrics['accuracy'])
        
        if 'f1_score' in metrics:
            self.model_f1_score.labels(
                model_name=model_name,
                model_version=model_version,
                dataset=dataset
            ).set(metrics['f1_score'])
        
        if 'mse' in metrics:
            self.model_mse.labels(
                model_name=model_name,
                model_version=model_version,
                dataset=dataset
            ).set(metrics['mse'])
    
    def update_deployment_metrics(self, deployments: Dict[str, Dict[str, Any]]) -> None:
        """Update deployment metrics."""
        self.model_deployments.set(len(deployments))
        
        for deployment_name, deployment_info in deployments.items():
            self.model_replicas.labels(
                deployment_name=deployment_name,
                model_name=deployment_info.get('model_name', 'unknown'),
                model_version=deployment_info.get('model_version', 'unknown')
            ).set(deployment_info.get('num_replicas_running', 0))
    
    def record_drift_detection(self, feature_name: str, drift_type: str) -> None:
        """Record drift detection event."""
        self.drift_detected.labels(
            feature_name=feature_name,
            drift_type=drift_type
        ).inc()
    
    def update_data_quality_score(self, dataset_name: str, score: float) -> None:
        """Update data quality score."""
        self.data_quality_score.labels(
            dataset_name=dataset_name
        ).set(score)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics in dict format."""
        try:
            # Generate latest metrics
            metrics_data = generate_latest(self.registry).decode('utf-8')
            
            # Parse metrics (simplified - in production, use proper parser)
            metrics = {}
            for line in metrics_data.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                
                try:
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = parts[1]
                        metrics[metric_name] = float(metric_value)
                except (ValueError, IndexError):
                    continue
            
            return metrics
            
        except Exception as e:
            self.log_error("Failed to get current metrics", error=str(e))
            return {}