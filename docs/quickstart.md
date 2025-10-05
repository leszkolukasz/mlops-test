# Quick Start Guide

Get up and running with the MLOps pipeline in minutes.

## Prerequisites

- Python 3.10 or higher
- Git
- Docker (optional, for deployment)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/leszkolukasz/mlops-test.git
cd mlops-test
```

### Step 2: Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Quick Run (Default Parameters)

```bash
python pipeline.py run
```

This will:
- Load the Iris dataset
- Engineer features
- Perform hyperparameter tuning (10 trials)
- Train the final model
- Log to MLflow
- Save model artifacts

**Expected output:**
```
Starting MLOps Pipeline
Feature Engineering Step
Loaded 150 samples
...
Pipeline Complete!
Test accuracy: 0.9667
```

### Option 2: Custom Parameters

```bash
python pipeline.py run \
  --test-size 0.3 \
  --val-size 0.2 \
  --n-trials 20
```

**Parameters:**
- `--test-size`: Test set size (default: 0.2)
- `--val-size`: Validation set size (default: 0.2)
- `--n-trials`: Number of hyperparameter tuning trials (default: 10)
- `--mlflow-uri`: MLflow tracking URI (default: file:./mlruns)

## Viewing Results

### MLflow UI

View experiment results and model registry:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

You'll see:
- All experiment runs
- Metrics comparison
- Model artifacts
- Parameters used

## Testing Individual Components

### Test Feature Engineering

```bash
python src/features/feature_engineering.py
```

**Output:**
```
Training set shape: (96, 8)
Test set shape: (24, 8)
Number of classes: 3
```

### Test Model Training

```bash
python src/models/train_model.py
```

**Output:**
```
Test metrics:
accuracy: 0.9667
precision: 0.9667
recall: 0.9667
f1_score: 0.9667
```

## Local Deployment Testing

### Ray Serve Deployment

```bash
# Start the server
python deployments/ray_serve/deploy.py

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Expected response:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "probabilities": [0.98, 0.01, 0.01]
}
```

### BentoML Deployment

```bash
# Save model to BentoML store
python deployments/bentoml/save_model.py

# Serve the model
cd deployments/bentoml
bentoml serve service:svc

# In another terminal, test
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## Docker Deployment

### Ray Serve

```bash
# Build the image
cd deployments/ray_serve
docker build -t iris-classifier-ray .

# Run the container
docker run -p 8000:8000 iris-classifier-ray

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @../../examples/example_request.json
```

### BentoML

```bash
cd deployments/bentoml

# Save model
python save_model.py

# Build Bento
bentoml build

# Containerize
bentoml containerize iris_classifier:latest

# Run
docker run -p 8001:3000 iris_classifier:latest

# Test
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d @../../examples/example_request.json
```

## Production Deployment

### Prerequisites

1. **Server with Docker**: A remote server with Docker installed
2. **SSH Access**: SSH key for authentication
3. **GitHub Secrets**: Configure the following secrets in your repository:
   - `DEPLOY_USER`: SSH username
   - `DEPLOY_HOST`: Server hostname/IP
   - `SSH_PRIVATE_KEY`: SSH private key

### Automatic Deployment via GitHub Actions

Push to main branch to trigger automatic deployment:

```bash
git add .
git commit -m "Update model"
git push origin main
```

The GitHub Actions workflow will:
1. Run the pipeline
2. Train the model
3. Deploy to both Ray Serve (port 8000) and BentoML (port 8001)

### Manual Deployment

If you need to deploy manually:

```bash
# Set environment variables
export DEPLOY_USER=your_username
export DEPLOY_HOST=your_server_ip
export SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)"

# Deploy Ray Serve
./deployments/ray_serve/deploy.sh

# Deploy BentoML
./deployments/bentoml/deploy.sh
```

## Troubleshooting

### Issue: Dependencies Installation Fails

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Issue: Ray Won't Start

**Solution:**
```bash
# Stop any existing Ray processes
ray stop

# Try again
python pipeline.py run
```

### Issue: MLflow UI Not Working

**Solution:**
```bash
# Specify the backend store URI explicitly
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

### Issue: Port Already in Use

**Solution:**
```bash
# Find and kill the process using the port
lsof -i :8000  # or :8001 for BentoML
kill -9 <PID>

# Or use a different port
python deployments/ray_serve/deploy.py --port 8080
```

### Issue: Model Files Not Found

**Solution:**
```bash
# Ensure you've run the pipeline first
python pipeline.py run

# Check if models exist
ls -la models/
```

## Next Steps

1. **Explore the Code**: Review the source code to understand the implementation
2. **Modify Features**: Try adding new features in `src/features/feature_engineering.py`
3. **Try Different Models**: Experiment with other scikit-learn models
4. **Adjust Hyperparameters**: Modify the search space in `pipeline.py`
5. **Compare Deployments**: Test both Ray Serve and BentoML to see differences
6. **Read Documentation**: Check `docs/` folder for detailed architecture and comparisons

## Common Use Cases

### Use Case 1: Experiment with Different Hyperparameters

```bash
# Run with more trials for better hyperparameters
python pipeline.py run --n-trials 50

# Check results in MLflow UI
mlflow ui
```

### Use Case 2: Deploy Multiple Model Versions

```bash
# Train version 1
python pipeline.py run

# Make changes to feature engineering or model
# ...

# Train version 2
python pipeline.py run

# Compare in MLflow UI
mlflow ui
```

### Use Case 3: A/B Testing

```bash
# Deploy version 1 to Ray Serve (port 8000)
# Deploy version 2 to BentoML (port 8001)
# Route traffic to both and compare performance
```

## Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **Metaflow Documentation**: https://docs.metaflow.org/
- **Ray Documentation**: https://docs.ray.io/
- **BentoML Documentation**: https://docs.bentoml.org/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the architecture documentation in `docs/`
3. Open an issue on GitHub
