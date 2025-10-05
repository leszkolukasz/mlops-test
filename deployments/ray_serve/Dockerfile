FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY deployments/ray_serve/ /app/deployments/ray_serve/
COPY models/ /app/models/

# Expose port
EXPOSE 8000

# Run the Ray Serve deployment
CMD ["python", "/app/deployments/ray_serve/deploy.py", "--model-path", "/app/models/iris_model.pkl", "--scaler-path", "/app/models/scaler.pkl"]
