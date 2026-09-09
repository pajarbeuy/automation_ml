FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/packages/ml_engine:/app/apps/api

# Copy configuration
COPY pyproject.toml /app/

# Copy packages and API source
COPY packages/ /app/packages/
COPY apps/api/ /app/apps/api/

# Install dependencies including MLflow, XGBoost, LightGBM
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir xgboost lightgbm mlflow

# Create storage directories
RUN mkdir -p /app/apps/api/storage/datasets /app/apps/api/storage/artifacts

WORKDIR /app/apps/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
