"""Model server with Ray Serve for scalable model serving."""

import ray
from ray import serve
from typing import Dict, Any, Optional, List
import asyncio
import time

from mlops_pipeline.utils.logger import LoggerMixin
from mlops_pipeline.utils.config import ServingConfig
from mlops_pipeline.serving.predictor import ModelPredictor, create_predictor_deployment


class ModelServer(LoggerMixin):
    """Model server manager for Ray Serve deployments."""
    
    def __init__(self, config: ServingConfig):
        super().__init__()
        self.config = config
        self.deployments = {}
        self._initialize_ray_serve()
    
    def _initialize_ray_serve(self) -> None:
        """Initialize Ray Serve."""
        self.log_info("Initializing Ray Serve")
        
        if not ray.is_initialized():
            ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
                ignore_reinit_error=True
            )
        
        serve.start(
            http_options={"host": self.config.host, "port": self.config.port},
            detached=True
        )
        
        self.log_info(
            "Ray Serve initialized",
            host=self.config.host,
            port=self.config.port
        )
    
    def deploy_model(
        self,
        model_name: str,
        deployment_name: Optional[str] = None,
        model_version: str = "latest",
        num_replicas: Optional[int] = None,
        max_concurrent_queries: Optional[int] = None,
        route_prefix: Optional[str] = None
    ) -> str:
        """Deploy a model for serving."""
        deployment_name = deployment_name or f"{model_name}-{model_version}"
        num_replicas = num_replicas or self.config.num_replicas
        max_concurrent_queries = max_concurrent_queries or self.config.max_concurrent_queries
        route_prefix = route_prefix or f"/{deployment_name}"
        
        self.log_info(
            "Deploying model",
            model_name=model_name,
            deployment_name=deployment_name,
            model_version=model_version,
            num_replicas=num_replicas,
            route_prefix=route_prefix
        )
        
        try:
            # Create deployment
            deployment = create_predictor_deployment(
                model_name=model_name,
                model_version=model_version,
                num_replicas=num_replicas,
                max_concurrent_queries=max_concurrent_queries
            )
            
            # Deploy with route prefix
            serve.run(
                deployment.bind(),
                name=deployment_name,
                route_prefix=route_prefix
            )
            
            # Store deployment info
            self.deployments[deployment_name] = {
                "model_name": model_name,
                "model_version": model_version,
                "num_replicas": num_replicas,
                "max_concurrent_queries": max_concurrent_queries,
                "route_prefix": route_prefix,
                "deployment_time": time.time()
            }
            
            self.log_info(
                "Model deployed successfully",
                deployment_name=deployment_name,
                route_prefix=route_prefix
            )
            
            return deployment_name
            
        except Exception as e:
            self.log_error(
                "Failed to deploy model",
                model_name=model_name,
                deployment_name=deployment_name,
                error=str(e)
            )
            raise
    
    def update_deployment(
        self,
        deployment_name: str,
        num_replicas: Optional[int] = None,
        max_concurrent_queries: Optional[int] = None
    ) -> None:
        """Update an existing deployment configuration."""
        if deployment_name not in self.deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        self.log_info(
            "Updating deployment",
            deployment_name=deployment_name,
            num_replicas=num_replicas,
            max_concurrent_queries=max_concurrent_queries
        )
        
        deployment_info = self.deployments[deployment_name]
        
        # Update configuration
        if num_replicas is not None:
            deployment_info["num_replicas"] = num_replicas
        if max_concurrent_queries is not None:
            deployment_info["max_concurrent_queries"] = max_concurrent_queries
        
        # Redeploy with new configuration
        deployment = create_predictor_deployment(
            model_name=deployment_info["model_name"],
            model_version=deployment_info["model_version"],
            num_replicas=deployment_info["num_replicas"],
            max_concurrent_queries=deployment_info["max_concurrent_queries"]
        )
        
        serve.run(
            deployment.bind(),
            name=deployment_name,
            route_prefix=deployment_info["route_prefix"]
        )
        
        self.log_info(
            "Deployment updated successfully",
            deployment_name=deployment_name
        )
    
    def undeploy_model(self, deployment_name: str) -> None:
        """Remove a model deployment."""
        self.log_info("Undeploying model", deployment_name=deployment_name)
        
        try:
            serve.delete(deployment_name)
            
            if deployment_name in self.deployments:
                del self.deployments[deployment_name]
            
            self.log_info(
                "Model undeployed successfully",
                deployment_name=deployment_name
            )
            
        except Exception as e:
            self.log_error(
                "Failed to undeploy model",
                deployment_name=deployment_name,
                error=str(e)
            )
            raise
    
    def list_deployments(self) -> Dict[str, Dict[str, Any]]:
        """List all active deployments."""
        self.log_info("Listing deployments")
        
        # Get Ray Serve deployments
        serve_deployments = serve.status().deployment_statuses
        
        # Combine with our tracked deployments
        all_deployments = {}
        
        for deployment_name, serve_status in serve_deployments.items():
            deployment_info = self.deployments.get(deployment_name, {})
            
            all_deployments[deployment_name] = {
                **deployment_info,
                "status": serve_status.status,
                "message": serve_status.message,
                "num_replicas_running": len([
                    r for r in serve_status.replica_states.values() 
                    if r.state == "RUNNING"
                ])
            }
        
        self.log_info(
            "Deployments listed",
            total_deployments=len(all_deployments)
        )
        
        return all_deployments
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get status of a specific deployment."""
        deployments = self.list_deployments()
        
        if deployment_name not in deployments:
            return {"error": f"Deployment {deployment_name} not found"}
        
        return deployments[deployment_name]
    
    def scale_deployment(self, deployment_name: str, num_replicas: int) -> None:
        """Scale a deployment to the specified number of replicas."""
        self.log_info(
            "Scaling deployment",
            deployment_name=deployment_name,
            num_replicas=num_replicas
        )
        
        self.update_deployment(
            deployment_name=deployment_name,
            num_replicas=num_replicas
        )
    
    def get_deployment_metrics(self, deployment_name: str) -> Dict[str, Any]:
        """Get metrics for a specific deployment."""
        # In a real implementation, this would collect metrics from Ray Serve
        # For now, return basic status information
        status = self.get_deployment_status(deployment_name)
        
        if "error" in status:
            return status
        
        return {
            "deployment_name": deployment_name,
            "status": status.get("status"),
            "num_replicas_running": status.get("num_replicas_running", 0),
            "num_replicas_target": status.get("num_replicas", 0),
            "uptime_seconds": time.time() - status.get("deployment_time", time.time()),
            "model_name": status.get("model_name"),
            "model_version": status.get("model_version")
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the model server."""
        try:
            serve_status = serve.status()
            deployments = self.list_deployments()
            
            healthy_deployments = sum(
                1 for d in deployments.values()
                if d.get("status") == "HEALTHY"
            )
            
            return {
                "status": "healthy" if serve_status.app_status.status == "RUNNING" else "unhealthy",
                "total_deployments": len(deployments),
                "healthy_deployments": healthy_deployments,
                "ray_serve_status": serve_status.app_status.status,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def shutdown(self) -> None:
        """Shutdown the model server."""
        self.log_info("Shutting down model server")
        
        # Undeploy all models
        for deployment_name in list(self.deployments.keys()):
            try:
                self.undeploy_model(deployment_name)
            except Exception as e:
                self.log_warning(
                    "Failed to undeploy model during shutdown",
                    deployment_name=deployment_name,
                    error=str(e)
                )
        
        # Shutdown Ray Serve
        try:
            serve.shutdown()
        except Exception as e:
            self.log_warning("Failed to shutdown Ray Serve", error=str(e))
        
        self.log_info("Model server shutdown completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()