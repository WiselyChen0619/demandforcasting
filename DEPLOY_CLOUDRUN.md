# Deploying Demand Forecasting API to Google Cloud Run

This guide explains how to deploy the Demand Forecasting API to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** installed (optional, for local testing)
4. **Project ID** from Google Cloud Console

## Quick Deployment

### Option 1: Using the Deploy Script

```bash
# Make script executable
chmod +x deploy.sh

# Run deployment (replace YOUR_PROJECT_ID)
./deploy.sh YOUR_PROJECT_ID asia-east1
```

### Option 2: Manual Deployment

1. **Set up Google Cloud**
```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

2. **Build and Deploy**
```bash
# Build container with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/demand-forecast-api .

# Deploy to Cloud Run
gcloud run deploy demand-forecast-api \
    --image gcr.io/YOUR_PROJECT_ID/demand-forecast-api \
    --platform managed \
    --region asia-east1 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --allow-unauthenticated
```

### Option 3: Using Cloud Build Trigger

1. Push code to GitHub
2. Set up Cloud Build trigger in Google Cloud Console
3. Use the provided `cloudbuild.yaml` for automatic deployments

## Local Testing

### 1. Build Docker Image Locally
```bash
docker build -t demand-forecast-api .
```

### 2. Run Container Locally
```bash
docker run -p 8080:8080 demand-forecast-api
```

### 3. Test the API
```bash
# Test with the provided script
python test_api.py http://localhost:8080

# Or use curl
curl http://localhost:8080/health
```

## API Endpoints

Once deployed, your API will be available at:
`https://demand-forecast-api-XXXXX-XX.a.run.app`

### Available Endpoints:

1. **GET /** - API information
2. **GET /health** - Health check
3. **GET /model_info** - Model information
4. **POST /predict** - Generate forecast
5. **POST /predict_sku** - SKU predictions

### Example Usage:

```bash
# Get 30-day forecast
curl -X POST https://YOUR_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_ahead": 30,
    "include_bounds": true
  }'

# Get top 10 SKU predictions
curl -X POST https://YOUR_URL/predict_sku \
  -H "Content-Type: application/json" \
  -d '{
    "top_n": 10,
    "days_ahead": 30
  }'
```

## Configuration

### Environment Variables
- `PORT`: Server port (default: 8080)
- `MODEL_PATH`: Path to model file (default: /app/models/demand_forecast_model.pkl)

### Resource Configuration
- **Memory**: 2GB (recommended for Prophet model)
- **CPU**: 2 vCPUs
- **Timeout**: 300 seconds
- **Max Instances**: 10 (adjust based on load)

## Cost Optimization

1. **Set maximum instances** to control costs
```bash
gcloud run services update demand-forecast-api --max-instances=5
```

2. **Enable CPU allocation only during requests**
```bash
gcloud run services update demand-forecast-api --cpu-throttling
```

3. **Set up Cloud Scheduler** for warming up the service
```bash
# Create a scheduler job to ping health endpoint every 5 minutes
gcloud scheduler jobs create http warm-up-forecast-api \
    --schedule="*/5 * * * *" \
    --uri="https://YOUR_URL/health" \
    --http-method=GET
```

## Monitoring

1. **View logs**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=demand-forecast-api" --limit 50
```

2. **Monitor metrics** in Cloud Console
- Go to Cloud Run > demand-forecast-api > Metrics
- Monitor: Request count, Latency, CPU usage, Memory usage

## Troubleshooting

### Common Issues:

1. **Memory Error**
   - Increase memory: `--memory 4Gi`
   - Reduce model complexity

2. **Timeout Error**
   - Increase timeout: `--timeout 600`
   - Optimize model loading

3. **Cold Start Issues**
   - Keep service warm with Cloud Scheduler
   - Reduce container size
   - Use minimum instances: `--min-instances 1`

### Debug Commands:

```bash
# Check service status
gcloud run services describe demand-forecast-api --region asia-east1

# View recent logs
gcloud logging read "resource.type=cloud_run_revision" --limit 20

# Test with verbose output
curl -v https://YOUR_URL/health
```

## Security

### 1. Enable Authentication (if needed)
```bash
# Remove --allow-unauthenticated flag
gcloud run deploy demand-forecast-api \
    --image gcr.io/YOUR_PROJECT_ID/demand-forecast-api \
    --platform managed \
    --region asia-east1
```

### 2. Use API Keys
Add API key validation in the application code

### 3. Set up Cloud Armor
Protect against DDoS and add IP restrictions

## CI/CD Integration

### GitHub Actions Example:
```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Build and Deploy
      run: |
        gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT_ID }}/demand-forecast-api
        gcloud run deploy demand-forecast-api \
          --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/demand-forecast-api \
          --platform managed \
          --region asia-east1 \
          --allow-unauthenticated
```

## Performance Tips

1. **Use multi-stage Docker builds** to reduce image size
2. **Cache model loading** in memory (already implemented)
3. **Use Cloud CDN** for static responses
4. **Implement request batching** for multiple predictions
5. **Use Cloud SQL** for model versioning

## Support

For issues or questions:
1. Check Cloud Run logs
2. Review error messages in API responses
3. Test locally with Docker first
4. Check resource limits and quotas