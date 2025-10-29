#!/bin/bash

# Azure App Service startup script
# This script starts the FastAPI application with Uvicorn

echo "Starting Hand-Written Equation Solver API..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Create necessary directories
mkdir -p logs uploads

# Start the application
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --access-log \
    --use-colors
