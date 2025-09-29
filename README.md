# MLOps Pipeline with Ray

A production-ready MLOps pipeline built with Ray for distributed computing, featuring comprehensive observability, model lifecycle management, and scalable serving infrastructure.

## 🚀 Features

### Core Components
- **Distributed Data Processing**: Ray Data for scalable data preprocessing and feature engineering
- **Model Training**: Ray Train for distributed training with MLflow integration
- **Hyperparameter Optimization**: Ray Tune for automated hyperparameter tuning
- **Model Serving**: Ray Serve for scalable model inference with auto-scaling
- **Model Registry**: MLflow-based model versioning and lifecycle management
- **Monitoring & Observability**: Prometheus metrics, structured logging, and drift detection
- **API Gateway**: FastAPI-based REST API for model interactions
- **Container Support**: Docker and Docker Compose for easy deployment

### Key Features
- **Distributed Computing**: Leverages Ray for horizontal scaling across multiple nodes
- **Production Ready**: Comprehensive logging, monitoring, and error handling
- **Model Lifecycle**: Full ML lifecycle from data ingestion to model retirement
- **Observability**: Built-in metrics collection, performance monitoring, and alerting
- **Scalable Serving**: Auto-scaling inference with load balancing
- **Data Quality**: Automated data validation and drift detection
- **Easy Deployment**: Docker containers and orchestration ready

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended for full pipeline

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/leszkolukasz/mlops-test.git
cd mlops-test
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Initialize Ray (optional)**:
```bash
ray start --head --dashboard-host=0.0.0.0
```

### Basic Usage

#### 1. Generate and Process Data
```bash
# Generate synthetic dataset
mlops-pipeline generate-data --samples 10000 --features 20 --task-type classification

# Process and split data
mlops-pipeline process-data --input data/raw/dataset.parquet --workers 4

# Validate data quality
mlops-pipeline validate-data --data-dir data/processed/
```

#### 2. Train Model
```bash
# Train with hyperparameter tuning
mlops-pipeline train-model --model-type random_forest --tune --max-trials 20

# Train with default parameters
mlops-pipeline train-model --model-type logistic_regression
```

#### 3. Serve Model
```bash
# Start model serving
mlops-pipeline serve-model --model-name mlops-model --replicas 3

# Start API server
mlops-pipeline start-api --host 0.0.0.0 --port 8080
```

#### 4. Monitor System
```bash
# Check system status
mlops-pipeline status

# Access Ray Dashboard
mlops-pipeline ray-dashboard
```

### Complete Example
```bash
# Run the full pipeline example
python examples/full_pipeline_example.py
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Training Layer │    │  Serving Layer  │
│                 │    │                 │    │                 │
│ • Data Generation│────▶• Ray Train     │────▶• Ray Serve     │
│ • Data Processing│    │ • Ray Tune     │    │ • FastAPI       │
│ • Data Validation│    │ • MLflow       │    │ • Auto-scaling  │
│ • Quality Checks │    │ • Model Registry│    │ • Load Balancing│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Monitoring Layer│
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • Drift Detection│
                    │ • Performance   │
                    └─────────────────┘
```

## 📊 Components Deep Dive

### Data Pipeline
- **Generator**: Creates synthetic datasets for testing and development
- **Processor**: Distributed data preprocessing with Ray Data
- **Validator**: Comprehensive data quality checks and validation
- **Features**: Automated feature engineering and scaling

### Training Pipeline
- **Trainer**: Distributed training with Ray Train
- **Tuner**: Automated hyperparameter optimization with Ray Tune
- **Registry**: Model versioning and lifecycle management with MLflow
- **Evaluation**: Comprehensive model evaluation and comparison

### Serving Pipeline
- **Model Server**: Scalable serving with Ray Serve
- **API Gateway**: REST API with FastAPI
- **Load Balancer**: Automatic load balancing and failover
- **Auto-scaling**: Dynamic scaling based on demand

### Monitoring Pipeline
- **Metrics Collection**: Prometheus-based metrics collection
- **Performance Monitoring**: Real-time performance tracking
- **Drift Detection**: Automated data and model drift detection
- **Alerting**: Configurable alerts and notifications

## 🔧 Configuration

The pipeline is configured through YAML files. Example configuration:

```yaml
# config/default.yaml
ray:
  dashboard_host: "0.0.0.0"
  dashboard_port: 8265

data:
  batch_size: 1000
  train_test_split: 0.8
  validation_split: 0.2

training:
  model_type: "random_forest"
  max_trials: 10
  model_registry_uri: "sqlite:///mlflow.db"

serving:
  host: "0.0.0.0"
  port: 8000
  num_replicas: 2

monitoring:
  metrics_port: 9090
  log_level: "INFO"
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose
```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Services Included
- **MLOps API**: Main application (port 8080)
- **Ray Serve**: Model serving (port 8000)
- **Ray Dashboard**: Cluster monitoring (port 8265)
- **MLflow**: Model registry (port 5000)
- **Prometheus**: Metrics collection (port 9091)
- **Grafana**: Monitoring dashboards (port 3000)

## 📈 Monitoring & Observability

### Metrics Available
- **System Metrics**: CPU, memory, disk usage
- **Ray Metrics**: Cluster resources, task execution
- **Model Metrics**: Prediction latency, throughput, accuracy
- **Data Metrics**: Quality scores, drift detection
- **Business Metrics**: Request rates, error rates

### Dashboards
- **Ray Dashboard**: http://localhost:8265
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **MLflow**: http://localhost:5000

### API Endpoints
- **Health Check**: `GET /health`
- **Predictions**: `POST /predict`
- **Model Management**: `GET /models`, `POST /models/deploy`
- **Metrics**: `GET /metrics`
- **Documentation**: `GET /docs`

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mlops_pipeline --cov-report=html

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## 📚 Examples

### Python API Usage
```python
from mlops_pipeline import Config
from mlops_pipeline.data import DataGenerator
from mlops_pipeline.training import ModelTrainer

# Load configuration
config = Config.from_yaml("config/default.yaml")

# Generate data
generator = DataGenerator(config.data)
df = generator.generate_classification_dataset(n_samples=1000)

# Train model
trainer = ModelTrainer(config.training)
model, metrics = trainer.train_model(X_train, y_train)
```

### REST API Usage
```bash
# Make predictions
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [{"feature_0": 1.0, "feature_1": 2.0}]}'

# Deploy new model
curl -X POST "http://localhost:8080/models/deploy" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "my-model", "model_version": "v1.0"}'

# Check system health
curl http://localhost:8080/health
```

## 🔍 Troubleshooting

### Common Issues

1. **Ray initialization fails**:
   ```bash
   ray stop  # Stop any existing Ray processes
   ray start --head --dashboard-host=0.0.0.0
   ```

2. **Memory issues**:
   - Reduce batch sizes in configuration
   - Limit number of Ray workers
   - Increase system memory or use distributed setup

3. **Port conflicts**:
   - Check for services running on default ports
   - Modify ports in configuration files
   - Use `mlops-pipeline status` to check service states

### Logs and Debugging
```bash
# Check application logs
tail -f logs/mlops-pipeline.log

# Check Ray logs
ray logs

# Enable debug logging
export LOG_LEVEL=DEBUG
mlops-pipeline --log-level DEBUG <command>
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone repository
git clone https://github.com/leszkolukasz/mlops-test.git
cd mlops-test

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ray](https://ray.io/) - Distributed computing framework
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Prometheus](https://prometheus.io/) - Monitoring and alerting
- [scikit-learn](https://scikit-learn.org/) - Machine learning library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/leszkolukasz/mlops-test/issues)
- **Discussions**: [GitHub Discussions](https://github.com/leszkolukasz/mlops-test/discussions)
- **Documentation**: [Project Wiki](https://github.com/leszkolukasz/mlops-test/wiki)

---

**Built with ❤️ for the MLOps community**