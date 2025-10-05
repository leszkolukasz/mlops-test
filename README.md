# MLOps Pipeline for Iris Classification

A comprehensive MLOps project demonstrating end-to-end machine learning pipeline with feature engineering, model training, hyperparameter tuning, model versioning, and deployment.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for Iris flower classification, featuring:

- **Feature Engineering**: Automated feature creation and preprocessing
- **Model Training**: Random Forest classifier with hyperparameter optimization
- **Hyperparameter Tuning**: Ray Tune integration for efficient search
- **Experiment Tracking**: MLflow for logging and model registry
- **Pipeline Orchestration**: Metaflow for workflow management
- **Deployment**: Two deployment options (Ray Serve and BentoML)
- **CI/CD**: GitHub Actions for automated training and deployment

## 📁 Project Structure

```
mlops-test/
├── src/
│   ├── features/
│   │   └── feature_engineering.py    # Feature engineering module
│   └── models/
│       └── train_model.py             # Model training module
├── deployments/
│   ├── ray_serve/
│   │   ├── deploy.py                  # Ray Serve deployment
│   │   ├── deploy.sh                  # Deployment script
│   │   └── Dockerfile                 # Docker configuration
│   └── bentoml/
│       ├── service.py                 # BentoML service definition
│       ├── save_model.py              # Model saving script
│       ├── deploy.sh                  # Deployment script
│       └── bentofile.yaml             # BentoML configuration
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml         # GitHub Actions workflow
├── config/
│   └── config.yaml                    # Configuration file
├── pipeline.py                        # Metaflow pipeline
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🚀 Features

### 1. Feature Engineering
- Loads Iris dataset from scikit-learn
- Creates engineered features:
  - Petal area (length × width)
  - Sepal area (length × width)
  - Petal/Sepal ratios
- Standardizes features using StandardScaler
- Saves scaler for deployment

### 2. Model Training with Hyperparameter Tuning
- Uses Random Forest Classifier
- Ray Tune for hyperparameter optimization:
  - Number of estimators
  - Maximum depth
  - Minimum samples split
  - Minimum samples leaf
- Optuna search algorithm for efficient exploration

### 3. MLflow Integration
- Experiment tracking
- Metric logging (accuracy, precision, recall, F1)
- Model versioning and registry
- Artifact storage

### 4. Metaflow Pipeline
Orchestrates the entire workflow:
1. **Start**: Initialize pipeline
2. **Feature Engineering**: Load and transform data
3. **Hyperparameter Tuning**: Find best parameters with Ray Tune
4. **Train Final Model**: Train with best parameters
5. **End**: Log results and save artifacts

### 5. Deployment Options

#### Ray Serve Deployment
- FastAPI-based REST API
- Horizontal scaling with replicas
- Async prediction support
- Health check endpoint

#### BentoML Deployment
- Optimized for ML model serving
- Built-in model versioning
- Containerization support
- Production-ready features

## 📋 Prerequisites

- Python 3.10+
- Docker (for deployment)
- SSH access to deployment server (for production)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/leszkolukasz/mlops-test.git
cd mlops-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Local Development

#### Run Feature Engineering
```bash
python src/features/feature_engineering.py
```

#### Run Model Training
```bash
python src/models/train_model.py
```

#### Run Complete Pipeline
```bash
python pipeline.py run
```

With custom parameters:
```bash
python pipeline.py run \
  --test-size 0.2 \
  --val-size 0.2 \
  --n-trials 20 \
  --mlflow-uri file:./mlruns
```

#### View MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Access at http://localhost:5000

### Local Deployment Testing

#### Ray Serve
```bash
python deployments/ray_serve/deploy.py \
  --model-path models/iris_model.pkl \
  --scaler-path models/scaler.pkl
```

Test the endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

#### BentoML
```bash
# Save model to BentoML store
python deployments/bentoml/save_model.py

# Serve the model
bentoml serve service:svc
```

Test the endpoint:
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **On Push/PR**:
   - Installs dependencies
   - Runs feature engineering tests
   - Runs model training tests
   - Executes Metaflow pipeline
   - Uploads model artifacts

2. **On Main Branch**:
   - Deploys to Ray Serve (port 8000)
   - Deploys to BentoML (port 8001)

### Required GitHub Secrets

Set these in your repository settings:
- `DEPLOY_USER`: SSH username for deployment server
- `DEPLOY_HOST`: Deployment server hostname/IP
- `SSH_PRIVATE_KEY`: SSH private key for authentication

### Manual Deployment Trigger

Use GitHub Actions workflow dispatch with deployment target selection:
- `ray`: Deploy only Ray Serve
- `bentoml`: Deploy only BentoML
- `both`: Deploy both services

## 🐳 Docker Deployment

### Ray Serve
```bash
cd deployments/ray_serve
docker build -t iris-classifier-ray .
docker run -p 8000:8000 iris-classifier-ray
```

### BentoML
```bash
cd deployments/bentoml
# First, save model and build
python save_model.py
bentoml build
bentoml containerize iris_classifier:latest
docker run -p 8001:3000 iris_classifier:latest
```

## 📊 Model Performance

The pipeline tracks multiple metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision (weighted average)
- **Recall**: Per-class recall (weighted average)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed per-class predictions

All metrics are logged to MLflow for comparison and analysis.

## 🔍 Monitoring and Observability

- **MLflow**: Track experiments, compare runs, visualize metrics
- **Model Registry**: Version control for models
- **Pipeline Cards**: Metaflow cards for run visualization
- **Health Checks**: Built-in endpoints for both deployments

## 🧪 Testing

Run individual components:
```bash
# Test feature engineering
python src/features/feature_engineering.py

# Test model training
python src/models/train_model.py

# Test full pipeline
python pipeline.py run --n-trials 5
```

## 📝 Configuration

Edit `config/config.yaml` to customize:
- MLflow tracking URI
- Hyperparameter search space
- Training parameters
- Deployment settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is open-source and available for educational purposes.

## 🎓 Learning Resources

This project demonstrates:
- **MLOps Best Practices**: Pipeline orchestration, experiment tracking, model versioning
- **Modern ML Tools**: Metaflow, Ray, MLflow, BentoML
- **Cloud-Native Deployment**: Docker, REST APIs, microservices
- **CI/CD for ML**: Automated training and deployment pipelines

## 🔮 Future Enhancements

- [ ] Add data validation with Great Expectations
- [ ] Implement A/B testing framework
- [ ] Add model monitoring and drift detection
- [ ] Support for more ML frameworks
- [ ] Kubernetes deployment configurations
- [ ] Feature store integration
- [ ] Model explainability (SHAP, LIME)
- [ ] Performance benchmarking dashboard

## 📧 Contact

For questions or feedback, please open an issue on GitHub.