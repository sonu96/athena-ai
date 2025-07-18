# Athena AI Deployment Guide

## Overview

This guide covers deploying Athena AI to Google Cloud Platform for 24/7 operation.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Docker installed locally
- CDP API keys
- OpenAI API key

## Local Development

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo>
cd athena-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your keys
```

### 2. Run Locally

```bash
# Run agent with API
python run.py

# Or run components separately
python main.py  # Agent only
python -m src.api.main  # API only
```

## Google Cloud Deployment

### 1. Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com
```

### 2. Create Secrets

```bash
# Use the setup script (recommended)
python scripts/setup_secrets.py

# Or create manually
echo -n "your-cdp-api-key" | gcloud secrets create cdp-api-key --data-file=-
echo -n "your-cdp-api-secret" | gcloud secrets create cdp-api-secret --data-file=-
echo -n "your-langsmith-api-key" | gcloud secrets create langsmith-api-key --data-file=-
echo -n "your-mem0-api-key" | gcloud secrets create mem0-api-key --data-file=-
```

### 3. Enable Google AI Platform

```bash
# Enable Vertex AI for Gemini
gcloud services enable aiplatform.googleapis.com

# Grant service account access to Vertex AI
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:YOUR-SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 4. Create Firestore Database

```bash
# Create Firestore database
gcloud firestore databases create \
  --location=us-central1 \
  --type=firestore-native
```

### 5. Build and Deploy

```bash
# Set project
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/athena-ai

# Deploy to Cloud Run
gcloud run deploy athena-ai \
  --image gcr.io/$PROJECT_ID/athena-ai \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 1 \
  --min-instances 1 \
  --port 8000 \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
  --set-secrets "CDP_API_KEY=cdp-api-key:latest,CDP_API_SECRET=cdp-api-secret:latest" \
  --allow-unauthenticated
```

### 6. Set Up Monitoring

```bash
# Create Cloud Scheduler jobs for maintenance
gcloud scheduler jobs create http claim-rewards \
  --location=us-central1 \
  --schedule="0 */4 * * *" \
  --uri="https://your-service-url/maintenance/claim-rewards" \
  --http-method=POST
```

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the application
CMD ["python", "run.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  athena:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CDP_API_KEY=${CDP_API_KEY}
      - CDP_API_SECRET=${CDP_API_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GCP_PROJECT_ID=${GCP_PROJECT_ID}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

## Production Configuration

### 1. Environment Variables

Set these in Cloud Run or your deployment platform:

```bash
# Required
GCP_PROJECT_ID=your-project-id
BASE_RPC_URL=https://mainnet.base.org

# Google AI
GOOGLE_AI_MODEL=gemini-1.5-pro
GOOGLE_LOCATION=us-central1

# Secrets (automatically loaded from Secret Manager)
# CDP_API_KEY, CDP_API_SECRET, LANGSMITH_API_KEY, MEM0_API_KEY
AGENT_CYCLE_TIME=300
AGENT_MAX_POSITION_SIZE=1000
LOG_LEVEL=INFO
```

### 2. Resource Allocation

Recommended Cloud Run settings:
- Memory: 2-4 GB
- CPU: 2 vCPUs
- Timeout: 3600 seconds
- Min instances: 1 (always on)
- Max instances: 1 (prevent multiple agents)

### 3. Security

- Use Secret Manager for all sensitive data
- Enable Cloud Armor for DDoS protection
- Set up Cloud IAM roles properly
- Use VPC connector for database access
- Enable audit logging

## Monitoring

### 1. Cloud Logging

View logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai" --limit 50
```

### 2. Metrics

Monitor key metrics:
- Agent uptime
- Reasoning cycles completed
- Total profit
- Error rate
- Memory usage

### 3. Alerts

Set up alerts for:
- Service downtime
- High error rate
- Low profit generation
- Memory issues

## Maintenance

### Daily Tasks
- Check logs for errors
- Verify agent is running
- Monitor profit generation

### Weekly Tasks
- Review memory usage
- Analyze strategy performance
- Check for new patterns discovered

### Monthly Tasks
- Rotate API keys
- Update dependencies
- Review and optimize costs
- Backup Firestore data

## Troubleshooting

### Agent Not Starting
1. Check logs: `gcloud logging read`
2. Verify secrets are set correctly
3. Ensure Firestore is accessible
4. Check CDP wallet initialization

### No Profits Generated
1. Verify Base RPC connection
2. Check wallet has funds
3. Review strategy configurations
4. Analyze gas prices

### Memory Issues
1. Check Firestore quotas
2. Verify Mem0 configuration
3. Review memory cleanup jobs
4. Monitor vector database size

## Cost Optimization

Estimated monthly costs:
- Cloud Run (always on): ~$50
- Firestore: ~$10-20
- Secret Manager: ~$1
- Cloud Logging: ~$5
- Total: ~$70-80/month

Tips:
- Use committed use discounts
- Set up budget alerts
- Monitor resource usage
- Optimize logging levels

## Backup and Recovery

### Backup Strategy
```bash
# Export Firestore data
gcloud firestore export gs://your-backup-bucket/$(date +%Y%m%d)

# Export memories
curl http://localhost:8000/memories/export > memories_backup.json
```

### Recovery Process
1. Deploy fresh instance
2. Import Firestore data
3. Restore wallet from ID
4. Resume operations

## Support

For issues:
1. Check logs first
2. Review this guide
3. Check API health endpoint
4. Contact support with error details

Remember: Athena learns continuously. The longer she runs, the better she performs!