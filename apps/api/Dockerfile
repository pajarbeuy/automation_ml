FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    sqlalchemy \
    alembic \
    psycopg2-binary \
    python-multipart \
    aiofiles \
    httpx

# Copy source
COPY . /app

# Add packages to PYTHONPATH
ENV PYTHONPATH=/app:/app/packages/ml_engine

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
