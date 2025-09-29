"""Structured logging utilities for the MLOps pipeline."""

import sys
import structlog
from typing import Any, Dict, Optional
from structlog.types import Processor


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    service_name: str = "mlops-pipeline",
) -> None:
    """Configure structured logging for the application."""
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Set logging level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with initial context."""
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **context)
    
    def log_error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **context)
    
    def log_warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **context)
    
    def log_debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **context)


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a logging context for function calls."""
    return {
        "function": func_name,
        "parameters": {k: str(v) for k, v in kwargs.items()},
    }


def log_performance_metrics(
    operation: str,
    duration_seconds: float,
    success: bool = True,
    **metrics: Any
) -> Dict[str, Any]:
    """Create a logging context for performance metrics."""
    return {
        "operation": operation,
        "duration_seconds": duration_seconds,
        "success": success,
        "metrics": metrics,
    }