"""Performance monitoring for models and system."""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from collections import deque, defaultdict
import statistics
import psutil
import ray

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import MonitoringConfig


class PerformanceMonitor(LoggerMixin):
    """Monitor system and model performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        super().__init__()
        self.config = config
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Metrics storage (in-memory for this example)
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.alert_callbacks = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'prediction_latency_ms': 1000.0,
            'error_rate': 0.05,
            'throughput_rps': 1.0  # minimum requests per second
        }
    
    def start_monitoring(self, interval_seconds: int = 10) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            self.log_warning("Performance monitoring already running")
            return
        
        self.log_info("Starting performance monitoring", interval_seconds=interval_seconds)
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.log_info("Stopping performance monitoring")
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect Ray metrics
                self._collect_ray_metrics()
                
                # Check thresholds and generate alerts
                self._check_alerts()
                
                # Wait for next interval
                self.stop_event.wait(interval_seconds)
                
            except Exception as e:
                self.log_error("Error in monitoring loop", error=str(e))
                self.stop_event.wait(interval_seconds)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        timestamp = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_history['cpu_usage'].append({
                'timestamp': timestamp,
                'value': cpu_percent
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_history['memory_usage'].append({
                'timestamp': timestamp,
                'value': memory.percent
            })
            
            self.metrics_history['memory_available'].append({
                'timestamp': timestamp,
                'value': memory.available / (1024**3)  # GB
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_history['disk_usage'].append({
                'timestamp': timestamp,
                'value': disk.percent
            })
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics_history['network_bytes_sent'].append({
                'timestamp': timestamp,
                'value': network.bytes_sent
            })
            
            self.metrics_history['network_bytes_recv'].append({
                'timestamp': timestamp,
                'value': network.bytes_recv
            })
            
            # Process metrics
            process = psutil.Process()
            self.metrics_history['process_cpu'].append({
                'timestamp': timestamp,
                'value': process.cpu_percent()
            })
            
            self.metrics_history['process_memory'].append({
                'timestamp': timestamp,
                'value': process.memory_info().rss / (1024**2)  # MB
            })
            
        except Exception as e:
            self.log_error("Failed to collect system metrics", error=str(e))
    
    def _collect_ray_metrics(self) -> None:
        """Collect Ray cluster metrics."""
        if not ray.is_initialized():
            return
        
        timestamp = time.time()
        
        try:
            # Cluster resources
            cluster_resources = ray.cluster_resources()
            cluster_state = ray.state.cluster_resources()
            
            # CPU metrics
            total_cpus = cluster_resources.get('CPU', 0)
            available_cpus = cluster_state.get('CPU', 0)
            used_cpus = total_cpus - available_cpus
            
            self.metrics_history['ray_cpus_total'].append({
                'timestamp': timestamp,
                'value': total_cpus
            })
            
            self.metrics_history['ray_cpus_used'].append({
                'timestamp': timestamp,
                'value': used_cpus
            })
            
            self.metrics_history['ray_cpu_utilization'].append({
                'timestamp': timestamp,
                'value': (used_cpus / total_cpus * 100) if total_cpus > 0 else 0
            })
            
            # Memory metrics
            total_memory = cluster_resources.get('memory', 0)
            available_memory = cluster_state.get('memory', 0)
            used_memory = total_memory - available_memory
            
            self.metrics_history['ray_memory_total'].append({
                'timestamp': timestamp,
                'value': total_memory / (1024**3)  # GB
            })
            
            self.metrics_history['ray_memory_used'].append({
                'timestamp': timestamp,
                'value': used_memory / (1024**3)  # GB
            })
            
            # Node metrics
            nodes = ray.nodes()
            alive_nodes = len([n for n in nodes if n['Alive']])
            
            self.metrics_history['ray_nodes_alive'].append({
                'timestamp': timestamp,
                'value': alive_nodes
            })
            
        except Exception as e:
            self.log_error("Failed to collect Ray metrics", error=str(e))
    
    def record_prediction_metrics(
        self,
        latency_ms: float,
        batch_size: int = 1,
        success: bool = True,
        model_name: str = "unknown"
    ) -> None:
        """Record prediction performance metrics."""
        timestamp = time.time()
        
        # Latency metrics
        self.metrics_history[f'prediction_latency_{model_name}'].append({
            'timestamp': timestamp,
            'value': latency_ms
        })
        
        # Throughput metrics (samples per second)
        throughput = batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0
        self.metrics_history[f'prediction_throughput_{model_name}'].append({
            'timestamp': timestamp,
            'value': throughput
        })
        
        # Success/error tracking
        self.metrics_history[f'prediction_success_{model_name}'].append({
            'timestamp': timestamp,
            'value': 1 if success else 0
        })
        
        # Batch size tracking
        self.metrics_history[f'prediction_batch_size_{model_name}'].append({
            'timestamp': timestamp,
            'value': batch_size
        })
    
    def _check_alerts(self) -> None:
        """Check metrics against thresholds and generate alerts."""
        current_time = time.time()
        
        # Check CPU usage
        if 'cpu_usage' in self.metrics_history:
            recent_cpu = [m['value'] for m in self.metrics_history['cpu_usage'] 
                         if current_time - m['timestamp'] < 300]  # Last 5 minutes
            if recent_cpu:
                avg_cpu = statistics.mean(recent_cpu)
                if avg_cpu > self.thresholds['cpu_usage']:
                    self._generate_alert(
                        'high_cpu_usage',
                        f"High CPU usage: {avg_cpu:.1f}%",
                        {'cpu_usage': avg_cpu, 'threshold': self.thresholds['cpu_usage']}
                    )
        
        # Check memory usage
        if 'memory_usage' in self.metrics_history:
            recent_memory = [m['value'] for m in self.metrics_history['memory_usage']
                           if current_time - m['timestamp'] < 300]
            if recent_memory:
                avg_memory = statistics.mean(recent_memory)
                if avg_memory > self.thresholds['memory_usage']:
                    self._generate_alert(
                        'high_memory_usage',
                        f"High memory usage: {avg_memory:.1f}%",
                        {'memory_usage': avg_memory, 'threshold': self.thresholds['memory_usage']}
                    )
        
        # Check disk usage
        if 'disk_usage' in self.metrics_history:
            recent_disk = [m['value'] for m in self.metrics_history['disk_usage']
                         if current_time - m['timestamp'] < 300]
            if recent_disk:
                avg_disk = statistics.mean(recent_disk)
                if avg_disk > self.thresholds['disk_usage']:
                    self._generate_alert(
                        'high_disk_usage',
                        f"High disk usage: {avg_disk:.1f}%",
                        {'disk_usage': avg_disk, 'threshold': self.thresholds['disk_usage']}
                    )
        
        # Check prediction latency for all models
        for metric_name in self.metrics_history:
            if metric_name.startswith('prediction_latency_'):
                model_name = metric_name.replace('prediction_latency_', '')
                recent_latency = [m['value'] for m in self.metrics_history[metric_name]
                                if current_time - m['timestamp'] < 300]
                if recent_latency:
                    avg_latency = statistics.mean(recent_latency)
                    if avg_latency > self.thresholds['prediction_latency_ms']:
                        self._generate_alert(
                            'high_prediction_latency',
                            f"High prediction latency for {model_name}: {avg_latency:.1f}ms",
                            {
                                'model_name': model_name,
                                'avg_latency_ms': avg_latency,
                                'threshold': self.thresholds['prediction_latency_ms']
                            }
                        )
    
    def _generate_alert(self, alert_type: str, message: str, metadata: Dict[str, Any]) -> None:
        """Generate an alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'metadata': metadata,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        self.log_warning(f"Alert generated: {message}", alert_type=alert_type, metadata=metadata)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.log_error("Alert callback failed", error=str(e))
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            'high_cpu_usage': 'warning',
            'high_memory_usage': 'warning',
            'high_disk_usage': 'critical',
            'high_prediction_latency': 'warning',
            'high_error_rate': 'critical',
            'low_throughput': 'warning'
        }
        return severity_map.get(alert_type, 'info')
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_time = time.time()
        current_metrics = {}
        
        for metric_name, metric_history in self.metrics_history.items():
            if not metric_history:
                continue
            
            # Get recent values (last 5 minutes)
            recent_values = [
                m['value'] for m in metric_history
                if current_time - m['timestamp'] < 300
            ]
            
            if recent_values:
                current_metrics[metric_name] = {
                    'current': recent_values[-1],
                    'avg_5min': statistics.mean(recent_values),
                    'min_5min': min(recent_values),
                    'max_5min': max(recent_values),
                    'count': len(recent_values)
                }
        
        return current_metrics
    
    def get_alerts(self, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get alerts since a specific timestamp."""
        if since_timestamp is None:
            return self.alerts.copy()
        
        return [
            alert for alert in self.alerts
            if alert['timestamp'] >= since_timestamp
        ]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        self.log_info("All alerts cleared")
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric."""
        self.thresholds[metric_name] = threshold
        self.log_info(f"Threshold updated", metric=metric_name, threshold=threshold)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        current_metrics = self.get_current_metrics()
        recent_alerts = self.get_alerts(time.time() - 3600)  # Last hour
        
        return {
            'timestamp': time.time(),
            'monitoring_active': self.is_monitoring,
            'metrics_collected': len(self.metrics_history),
            'current_metrics': current_metrics,
            'recent_alerts': len(recent_alerts),
            'alert_summary': self._get_alert_summary(recent_alerts),
            'thresholds': self.thresholds.copy()
        }
    
    def _get_alert_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get alert summary by type and severity."""
        summary = defaultdict(int)
        
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            severity = alert.get('severity', 'info')
            
            summary[f'type_{alert_type}'] += 1
            summary[f'severity_{severity}'] += 1
        
        return dict(summary)
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()