#!/bin/bash

# Deployment script for Google Cloud Run
# Usage: ./deploy.sh [PROJECT_ID] [REGION]

set -e

# Default values
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"asia-east1"}
SERVICE_NAME="demand-forecast-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Deploying Demand Forecast API to Cloud Run ==="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    exit 1
fi

# Set project
echo "Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build container using Cloud Build
echo "Building container with Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --allow-unauthenticated \
    --set-env-vars="MODEL_PATH=/app/models/demand_forecast_model.pkl"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo
echo "=== Deployment Complete ==="
echo "Service URL: ${SERVICE_URL}"
echo
echo "Test endpoints:"
echo "  Health: curl ${SERVICE_URL}/health"
echo "  Predict: curl -X POST ${SERVICE_URL}/predict -H 'Content-Type: application/json' -d '{\"days_ahead\": 7}'"
echo "  Model Info: curl ${SERVICE_URL}/model_info"