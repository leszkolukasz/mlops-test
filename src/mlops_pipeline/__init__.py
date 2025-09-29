"""MLOps Pipeline with Ray for distributed computing and observability."""

__version__ = "0.1.0"
__author__ = "MLOps Pipeline Team"
__email__ = "mlops@example.com"

from mlops_pipeline.utils.config import Config
from mlops_pipeline.utils.logger import get_logger

__all__ = ["Config", "get_logger"]