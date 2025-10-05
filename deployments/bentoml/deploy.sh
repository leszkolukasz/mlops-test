#!/bin/bash
# Deploy BentoML model to remote server

set -e

# Configuration
REMOTE_USER="${DEPLOY_USER}"
REMOTE_HOST="${DEPLOY_HOST}"
REMOTE_PATH="/opt/iris-classifier-bento"
SSH_KEY="${SSH_PRIVATE_KEY}"

echo "Deploying BentoML model to ${REMOTE_HOST}..."

# Create SSH key file
mkdir -p ~/.ssh
echo "${SSH_KEY}" > ~/.ssh/deploy_key
chmod 600 ~/.ssh/deploy_key

# Create remote directory
ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# Copy files to remote server
echo "Copying files..."
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no -r deployments/bentoml/* ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no -r models/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no requirements.txt ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# Setup and deploy BentoML
echo "Setting up BentoML..."
ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd /opt/iris-classifier-bento

# Install dependencies
pip install bentoml scikit-learn joblib numpy

# Save model to BentoML store
python save_model.py --model-path models/iris_model.pkl --scaler-path models/scaler.pkl

# Build Bento
bentoml build

# Get the latest bento tag
BENTO_TAG=$(bentoml list iris_classifier -o json | jq -r '.[0].tag')

# Build Docker image
bentoml containerize ${BENTO_TAG} -t iris-classifier-bento:latest

# Stop existing container if running
docker stop iris-classifier-bento || true
docker rm iris-classifier-bento || true

# Run new container
docker run -d \
  --name iris-classifier-bento \
  --restart unless-stopped \
  -p 8001:3000 \
  iris-classifier-bento:latest

echo "BentoML deployment completed!"
docker ps | grep iris-classifier-bento
EOF

echo "Deployment successful! Model is available at http://${REMOTE_HOST}:8001"
echo "Test with: curl -X POST http://${REMOTE_HOST}:8001/predict -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'"
