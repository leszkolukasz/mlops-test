"""FastAPI application for model serving API."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import time
import httpx
from contextlib import asynccontextmanager

from mlops_pipeline.utils.logger import LoggerMixin, get_logger
from mlops_pipeline.utils.config import ServingConfig
from mlops_pipeline.serving.model_server import ModelServer
from mlops_pipeline.monitoring.metrics_collector import MetricsCollector


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., description="Input data for prediction"
    )
    model_name: Optional[str] = Field(
        None, description="Specific model name to use for prediction"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[str]] = None
    model_name: str
    model_version: str
    prediction_time_ms: float
    num_samples: int


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class DeploymentRequest(BaseModel):
    """Request model for model deployment."""
    model_name: str
    model_version: str = "latest"
    num_replicas: int = Field(default=2, ge=1, le=10)
    route_prefix: Optional[str] = None


class DeploymentResponse(BaseModel):
    """Response model for deployment operations."""
    deployment_name: str
    status: str
    message: Optional[str] = None


# Global variables (in production, use dependency injection)
model_server: Optional[ModelServer] = None
metrics_collector: Optional[MetricsCollector] = None
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_server, metrics_collector
    
    # Startup
    logger.info("Starting MLOps API server")
    
    config = ServingConfig()
    model_server = ModelServer(config)
    
    # Initialize metrics collector
    try:
        from mlops_pipeline.monitoring.metrics_collector import MetricsCollector
        from mlops_pipeline.utils.config import MonitoringConfig
        monitoring_config = MonitoringConfig()
        metrics_collector = MetricsCollector(monitoring_config)
        metrics_collector.start_metrics_server()
    except Exception as e:
        logger.warning("Failed to start metrics collector", error=str(e))
    
    logger.info("MLOps API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MLOps API server")
    
    if model_server:
        model_server.shutdown()
    
    if metrics_collector:
        metrics_collector.stop_metrics_server()
    
    logger.info("MLOps API server shutdown completed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="MLOps Pipeline API",
        description="Production-ready MLOps pipeline with Ray for distributed computing",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "MLOps Pipeline API",
            "version": "0.1.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            if model_server:
                health_info = model_server.health_check()
                return HealthResponse(
                    status=health_info["status"],
                    timestamp=health_info["timestamp"],
                    details=health_info
                )
            else:
                return HealthResponse(
                    status="unhealthy",
                    timestamp=time.time(),
                    details={"error": "Model server not initialized"}
                )
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make predictions using deployed models."""
        start_time = time.time()
        
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            # Get deployments
            deployments = model_server.list_deployments()
            
            if not deployments:
                raise HTTPException(status_code=404, detail="No models deployed")
            
            # Select deployment
            if request.model_name:
                deployment_name = None
                for name, info in deployments.items():
                    if info.get("model_name") == request.model_name:
                        deployment_name = name
                        break
                
                if not deployment_name:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {request.model_name} not found"
                    )
            else:
                # Use first available deployment
                deployment_name = list(deployments.keys())[0]
            
            deployment_info = deployments[deployment_name]
            route_prefix = deployment_info.get("route_prefix", f"/{deployment_name}")
            
            # Make request to Ray Serve deployment
            base_url = f"http://{model_server.config.host}:{model_server.config.port}"
            url = f"{base_url}{route_prefix}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=request.data)
                response.raise_for_status()
                result = response.json()
            
            # Record metrics
            if metrics_collector:
                prediction_time = (time.time() - start_time) * 1000
                metrics_collector.record_prediction(
                    model_name=deployment_info.get("model_name", "unknown"),
                    model_version=deployment_info.get("model_version", "unknown"),
                    prediction_time_ms=prediction_time,
                    num_samples=result.get("num_samples", 1),
                    success=True
                )
            
            return PredictionResponse(**result)
            
        except HTTPException:
            raise
        except Exception as e:
            # Record error metrics
            if metrics_collector:
                prediction_time = (time.time() - start_time) * 1000
                metrics_collector.record_prediction(
                    model_name=request.model_name or "unknown",
                    model_version="unknown",
                    prediction_time_ms=prediction_time,
                    num_samples=1,
                    success=False
                )
            
            logger.error("Prediction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models", response_model=Dict[str, Any])
    async def list_models():
        """List all deployed models."""
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            deployments = model_server.list_deployments()
            return {"deployments": deployments}
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/deploy", response_model=DeploymentResponse)
    async def deploy_model(request: DeploymentRequest, background_tasks: BackgroundTasks):
        """Deploy a model for serving."""
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            # Deploy model in background
            def deploy():
                try:
                    deployment_name = model_server.deploy_model(
                        model_name=request.model_name,
                        model_version=request.model_version,
                        num_replicas=request.num_replicas,
                        route_prefix=request.route_prefix
                    )
                    logger.info("Model deployed successfully", deployment_name=deployment_name)
                except Exception as e:
                    logger.error("Failed to deploy model", error=str(e))
            
            background_tasks.add_task(deploy)
            
            return DeploymentResponse(
                deployment_name=f"{request.model_name}-{request.model_version}",
                status="deploying",
                message="Deployment started in background"
            )
            
        except Exception as e:
            logger.error("Failed to start deployment", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/models/{deployment_name}", response_model=DeploymentResponse)
    async def undeploy_model(deployment_name: str):
        """Undeploy a model."""
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            model_server.undeploy_model(deployment_name)
            
            return DeploymentResponse(
                deployment_name=deployment_name,
                status="undeployed",
                message="Model undeployed successfully"
            )
            
        except Exception as e:
            logger.error("Failed to undeploy model", deployment_name=deployment_name, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/{deployment_name}/status", response_model=Dict[str, Any])
    async def get_deployment_status(deployment_name: str):
        """Get status of a specific deployment."""
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            status = model_server.get_deployment_status(deployment_name)
            
            if "error" in status:
                raise HTTPException(status_code=404, detail=status["error"])
            
            return status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get deployment status", deployment_name=deployment_name, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/models/{deployment_name}/scale", response_model=DeploymentResponse)
    async def scale_deployment(deployment_name: str, num_replicas: int):
        """Scale a deployment to the specified number of replicas."""
        try:
            if not model_server:
                raise HTTPException(status_code=503, detail="Model server not available")
            
            if num_replicas < 1 or num_replicas > 10:
                raise HTTPException(
                    status_code=400,
                    detail="Number of replicas must be between 1 and 10"
                )
            
            model_server.scale_deployment(deployment_name, num_replicas)
            
            return DeploymentResponse(
                deployment_name=deployment_name,
                status="scaled",
                message=f"Deployment scaled to {num_replicas} replicas"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to scale deployment", deployment_name=deployment_name, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics", response_model=Dict[str, Any])
    async def get_metrics():
        """Get system and model metrics."""
        try:
            if not metrics_collector:
                raise HTTPException(status_code=503, detail="Metrics collector not available")
            
            metrics = metrics_collector.get_current_metrics()
            return {"metrics": metrics}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create app instance
app = create_app()