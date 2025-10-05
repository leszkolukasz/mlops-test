# MLOps Pipeline Architecture

This document describes the architecture and components of the MLOps pipeline.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Source   │  │   Config    │  │  Workflows   │  │  Deploy    │ │
│  │   Code     │  │   Files     │  │  (Actions)   │  │  Scripts   │ │
│  └────────────┘  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Push/PR Trigger
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GitHub Actions Runner                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Metaflow Pipeline                           │ │
│  │                                                                │ │
│  │  ┌──────────────┐                                             │ │
│  │  │    Start     │                                             │ │
│  │  └──────┬───────┘                                             │ │
│  │         │                                                     │ │
│  │         ▼                                                     │ │
│  │  ┌──────────────────┐                                        │ │
│  │  │ Feature Engineer │                                        │ │
│  │  │  - Load Data     │                                        │ │
│  │  │  - Create Feats  │                                        │ │
│  │  │  - Scale Data    │                                        │ │
│  │  └──────┬───────────┘                                        │ │
│  │         │                                                     │ │
│  │         ▼                                                     │ │
│  │  ┌──────────────────────┐                                    │ │
│  │  │ Hyperparameter Tune  │                                    │ │
│  │  │  ┌────────────────┐  │                                    │ │
│  │  │  │   Ray Tune     │  │                                    │ │
│  │  │  │ ┌────────────┐ │  │                                    │ │
│  │  │  │ │  Optuna    │ │  │                                    │ │
│  │  │  │ │  Search    │ │  │                                    │ │
│  │  │  │ └────────────┘ │  │                                    │ │
│  │  │  └────────────────┘  │                                    │ │
│  │  └──────┬───────────────┘                                    │ │
│  │         │                                                     │ │
│  │         ▼                                                     │ │
│  │  ┌──────────────────┐                                        │ │
│  │  │  Train Final     │                                        │ │
│  │  │  Model           │                                        │ │
│  │  │  - RF Classifier │                                        │ │
│  │  │  - Evaluate      │                                        │ │
│  │  └──────┬───────────┘                                        │ │
│  │         │                                                     │ │
│  │         ▼                                                     │ │
│  │  ┌──────────────────┐                                        │ │
│  │  │      End         │                                        │ │
│  │  │  - Save Model    │                                        │ │
│  │  │  - Save Metrics  │                                        │ │
│  │  └──────────────────┘                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      MLflow                                 │  │
│  │  - Experiment Tracking                                      │  │
│  │  - Metrics Logging                                          │  │
│  │  - Model Registry                                           │  │
│  │  - Artifact Storage                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              │ Artifacts                           │
│                              ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            Upload Model Artifacts                           │  │
│  │  - iris_model.pkl                                          │  │
│  │  - scaler.pkl                                              │  │
│  │  - mlruns/                                                 │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  │                                    │
                  ▼                                    ▼
    ┌──────────────────────────┐        ┌──────────────────────────┐
    │  Deploy Ray Serve Job    │        │  Deploy BentoML Job      │
    └────────────┬─────────────┘        └────────────┬─────────────┘
                 │                                    │
                 │ SSH Deploy                         │ SSH Deploy
                 ▼                                    ▼
    ┌──────────────────────────┐        ┌──────────────────────────┐
    │   Production Server      │        │   Production Server      │
    │                          │        │                          │
    │  ┌────────────────────┐  │        │  ┌────────────────────┐  │
    │  │  Docker Container  │  │        │  │  Docker Container  │  │
    │  │                    │  │        │  │                    │  │
    │  │  ┌──────────────┐  │  │        │  │  ┌──────────────┐  │  │
    │  │  │  Ray Serve   │  │  │        │  │  │  BentoML     │  │  │
    │  │  │              │  │  │        │  │  │  Service     │  │  │
    │  │  │  Port: 8000  │  │  │        │  │  │  Port: 8001  │  │  │
    │  │  └──────────────┘  │  │        │  │  └──────────────┘  │  │
    │  └────────────────────┘  │        │  └────────────────────┘  │
    └──────────────────────────┘        └──────────────────────────┘
                 │                                    │
                 │                                    │
                 ▼                                    ▼
    ┌──────────────────────────┐        ┌──────────────────────────┐
    │   Client Applications    │        │   Client Applications    │
    │   POST /predict          │        │   POST /predict          │
    │   GET  /health           │        │   GET  /health           │
    └──────────────────────────┘        └──────────────────────────┘
```

## Component Details

### 1. Source Code (`src/`)

#### Feature Engineering (`src/features/feature_engineering.py`)
- **Purpose**: Data loading and feature creation
- **Key Functions**:
  - `load_data()`: Load Iris dataset
  - `create_features()`: Engineer new features
  - `prepare_data()`: Scale and prepare for training
- **Output**: Processed training and test data

#### Model Training (`src/models/train_model.py`)
- **Purpose**: Model training and evaluation
- **Key Functions**:
  - `train()`: Train Random Forest classifier
  - `evaluate()`: Calculate metrics
  - `save_model()`: Persist model
- **Output**: Trained model and metrics

### 2. Pipeline Orchestration (`pipeline.py`)

**Metaflow Flow** orchestrates the entire pipeline:

1. **Start Step**
   - Initialize MLflow
   - Set experiment
   - Configure parameters

2. **Feature Engineering Step**
   - Load and transform data
   - Create train/val/test splits
   - Save feature scaler

3. **Hyperparameter Tuning Step**
   - Initialize Ray cluster
   - Define search space
   - Run Optuna search with Ray Tune
   - Select best parameters

4. **Train Final Model Step**
   - Train with best parameters
   - Evaluate on validation and test sets
   - Log metrics to MLflow
   - Register model in MLflow

5. **End Step**
   - Summary and cleanup
   - Display results

### 3. Experiment Tracking (MLflow)

**Components**:
- **Tracking Server**: Logs experiments
- **Model Registry**: Versions models
- **Artifact Store**: Stores model files

**Logged Information**:
- Hyperparameters
- Metrics (accuracy, precision, recall, F1)
- Model artifacts
- Confusion matrices
- Feature scaler

### 4. Deployment Options

#### Ray Serve Deployment

**Architecture**:
```
FastAPI Application
    │
    ├─ POST /predict
    │   └─ IrisClassifier.predict()
    │       ├─ Feature Engineering
    │       ├─ Scaling
    │       └─ Model Prediction
    │
    └─ GET /health
        └─ Health check
```

**Features**:
- 2 replicas for load balancing
- 0.5 CPU per replica
- Async prediction support
- FastAPI-based REST API

#### BentoML Deployment

**Architecture**:
```
BentoML Service
    │
    ├─ POST /predict
    │   └─ predict()
    │       ├─ Feature Engineering
    │       ├─ Scaling
    │       ├─ Adaptive Batching
    │       └─ Model Prediction
    │
    └─ GET /health
        └─ Health check
```

**Features**:
- Built-in model versioning
- Adaptive batching
- Containerization support
- Model store integration

### 5. CI/CD Pipeline (GitHub Actions)

**Workflow Jobs**:

1. **feature-engineering-and-training**
   - Install dependencies
   - Run tests
   - Execute Metaflow pipeline
   - Upload artifacts

2. **deploy-ray-serve**
   - Download model artifacts
   - SSH to production server
   - Build Docker image
   - Deploy container

3. **deploy-bentoml**
   - Download model artifacts
   - SSH to production server
   - Build Bento
   - Deploy container

4. **notify**
   - Check status
   - Report results

## Data Flow

1. **Training Phase**:
   ```
   Raw Data → Feature Engineering → Train/Val/Test Split → 
   Hyperparameter Tuning → Final Training → Model Artifacts
   ```

2. **Inference Phase**:
   ```
   Client Request → API Endpoint → Feature Engineering → 
   Scaling → Model Prediction → Response
   ```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Metaflow | Pipeline management |
| Hyperparameter Tuning | Ray Tune + Optuna | Efficient search |
| Experiment Tracking | MLflow | Logging and versioning |
| Model Framework | Scikit-learn | ML algorithms |
| Deployment (Option 1) | Ray Serve | Model serving |
| Deployment (Option 2) | BentoML | Model serving |
| API Framework | FastAPI | REST endpoints |
| Containerization | Docker | Application packaging |
| CI/CD | GitHub Actions | Automation |

## Security Considerations

1. **Secrets Management**:
   - SSH keys stored in GitHub Secrets
   - No credentials in code

2. **Network Security**:
   - Services run in Docker containers
   - Configurable ports
   - Health check endpoints

3. **Model Security**:
   - Model versioning for rollback
   - Artifact validation
   - Access control on deployment

## Scalability

### Horizontal Scaling
- **Ray Serve**: Adjust `num_replicas`
- **BentoML**: Adjust worker count
- Load balancer for multiple instances

### Vertical Scaling
- Adjust CPU/memory per replica
- Configure resource limits in Docker

### Auto-scaling
- Can be integrated with Kubernetes HPA
- Metrics-based scaling policies

## Monitoring

### Application Monitoring
- Health check endpoints
- Request/response logging
- Error tracking

### Model Monitoring
- Prediction distribution
- Inference latency
- Model accuracy over time

### Infrastructure Monitoring
- Container health
- Resource usage
- Network metrics

## Future Enhancements

1. **Data Validation**: Add Great Expectations
2. **Feature Store**: Integrate Feast or Hopsworks
3. **Model Monitoring**: Add drift detection
4. **A/B Testing**: Compare model versions
5. **Kubernetes**: Deploy to K8s cluster
6. **Observability**: Add Prometheus and Grafana
7. **Model Explainability**: Integrate SHAP/LIME
