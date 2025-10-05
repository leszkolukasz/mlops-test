#!/bin/bash
# Deploy Ray Serve model to remote server

set -e

# Configuration
REMOTE_USER="${DEPLOY_USER}"
REMOTE_HOST="${DEPLOY_HOST}"
REMOTE_PATH="/opt/iris-classifier-ray"
SSH_KEY="${SSH_PRIVATE_KEY}"

echo "Deploying Ray Serve model to ${REMOTE_HOST}..."

# Create SSH key file
mkdir -p ~/.ssh
echo "${SSH_KEY}" > ~/.ssh/deploy_key
chmod 600 ~/.ssh/deploy_key

# Create remote directory
ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# Copy files to remote server
echo "Copying files..."
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no -r deployments/ray_serve/* ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no -r models/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no -r src/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no requirements.txt ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# Build and run Docker container
echo "Building Docker image..."
ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd /opt/iris-classifier-ray
docker build -t iris-classifier-ray:latest -f Dockerfile .

# Stop existing container if running
docker stop iris-classifier-ray || true
docker rm iris-classifier-ray || true

# Run new container
docker run -d \
  --name iris-classifier-ray \
  --restart unless-stopped \
  -p 8000:8000 \
  iris-classifier-ray:latest

echo "Ray Serve deployment completed!"
docker ps | grep iris-classifier-ray
EOF

echo "Deployment successful! Model is available at http://${REMOTE_HOST}:8000"
echo "Test with: curl -X POST http://${REMOTE_HOST}:8000/predict -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'"
