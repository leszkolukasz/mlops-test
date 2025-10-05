#!/bin/bash
# Test deployment script - for local testing before deploying to remote server

set -e

echo "Testing deployments locally..."

# Check if models exist
if [ ! -f "models/iris_model.pkl" ]; then
    echo "❌ Model not found. Run the pipeline first:"
    echo "   python pipeline.py run"
    exit 1
fi

echo "✅ Model found"

# Test Ray Serve deployment
echo ""
echo "Testing Ray Serve deployment..."
echo "Starting Ray Serve in background..."

# Start Ray Serve
python deployments/ray_serve/deploy.py --model-path models/iris_model.pkl --scaler-path models/scaler.pkl &
RAY_PID=$!

# Wait for service to start
sleep 10

# Test the endpoint
echo "Testing Ray Serve endpoint..."
response=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}')

echo "Response: $response"

if echo "$response" | grep -q "prediction"; then
    echo "✅ Ray Serve deployment successful"
else
    echo "❌ Ray Serve deployment failed"
fi

# Cleanup
kill $RAY_PID 2>/dev/null || true

echo ""
echo "Testing complete!"
echo ""
echo "To deploy to production server, set these environment variables:"
echo "  - DEPLOY_USER"
echo "  - DEPLOY_HOST"
echo "  - SSH_PRIVATE_KEY"
echo ""
echo "Then run:"
echo "  ./deployments/ray_serve/deploy.sh (for Ray Serve)"
echo "  ./deployments/bentoml/deploy.sh (for BentoML)"
