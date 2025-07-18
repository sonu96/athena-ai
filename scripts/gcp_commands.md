# Google Cloud Commands for Athena AI

## Quick Setup Commands

### 1. Initial Setup (Run these in order)
```bash
# Login to Google Cloud
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Run the setup script
./scripts/gcp_setup.sh

# Configure secrets
python scripts/setup_secrets.py
```

## Project Management Commands

### List all projects
```bash
gcloud projects list
```

### Delete specific projects
```bash
# Delete a single project
gcloud projects delete PROJECT_ID

# Delete multiple projects
for project in project1 project2 project3; do
  gcloud projects delete $project --quiet
done
```

### Clean up old projects interactively
```bash
./scripts/gcp_cleanup.sh
```

### Set active project
```bash
gcloud config set project YOUR_PROJECT_ID
```

## Deployment Commands

### Deploy to Cloud Run
```bash
# Build and deploy in one command
gcloud run deploy athena-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 1 \
  --min-instances 1 \
  --set-env-vars "GCP_PROJECT_ID=$(gcloud config get-value project)"
```

### Deploy with Cloud Build
```bash
# Submit build
gcloud builds submit --config deployment/cloudbuild.yaml

# Deploy the built image
gcloud run deploy athena-ai \
  --image gcr.io/PROJECT_ID/athena-ai \
  --platform managed \
  --region us-central1
```

## Secret Management

### Create secrets manually
```bash
# Create a secret
echo -n "your-secret-value" | gcloud secrets create SECRET_NAME --data-file=-

# Update a secret
echo -n "new-secret-value" | gcloud secrets versions add SECRET_NAME --data-file=-

# List secrets
gcloud secrets list

# View secret value (be careful!)
gcloud secrets versions access latest --secret=SECRET_NAME
```

### Grant access to secrets
```bash
# For Cloud Run service
gcloud secrets add-iam-policy-binding SECRET_NAME \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Monitoring Commands

### View logs
```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai" --limit 50

# All logs from today
gcloud logging read "timestamp>=\"$(date -u +%Y-%m-%d)\"" --limit 100

# Error logs only
gcloud logging read "severity>=ERROR" --limit 50

# Follow logs (tail -f equivalent)
gcloud alpha logging tail "resource.type=cloud_run_revision"
```

### Check service status
```bash
# List Cloud Run services
gcloud run services list

# Describe specific service
gcloud run services describe athena-ai --region us-central1

# Get service URL
gcloud run services describe athena-ai --region us-central1 --format='value(status.url)'
```

## Cost Management

### View current billing
```bash
# List billing accounts
gcloud billing accounts list

# Check project billing
gcloud billing projects describe $(gcloud config get-value project)

# Set budget alerts (via console recommended)
echo "Set up budget alerts at: https://console.cloud.google.com/billing/budgets"
```

### Disable expensive services
```bash
# Disable specific APIs to save costs
gcloud services disable aiplatform.googleapis.com --force
gcloud services disable run.googleapis.com --force

# Or disable entire project (keeps data)
gcloud projects update PROJECT_ID --no-enable-apis
```

## Cleanup Commands

### Complete project cleanup
```bash
# Delete all resources and project
PROJECT_ID=your-project-id

# Delete Cloud Run services
gcloud run services list --format="value(name)" | xargs -I {} gcloud run services delete {} --quiet

# Delete secrets
gcloud secrets list --format="value(name)" | xargs -I {} gcloud secrets delete {} --quiet

# Delete Firestore data
gcloud firestore databases delete --quiet

# Finally, delete the project
gcloud projects delete $PROJECT_ID --quiet
```

### Resource-specific cleanup
```bash
# Delete old Cloud Run revisions
gcloud run revisions list --service athena-ai --format="value(name)" | tail -n +2 | xargs -I {} gcloud run revisions delete {} --quiet

# Clean up container images
gcloud container images list --format="value(name)" | xargs -I {} gcloud container images delete {} --quiet
```

## Useful Aliases

Add these to your `.bashrc` or `.zshrc`:

```bash
# Athena AI shortcuts
alias athena-logs='gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai" --limit 50'
alias athena-deploy='gcloud run deploy athena-ai --source . --region us-central1'
alias athena-url='gcloud run services describe athena-ai --region us-central1 --format="value(status.url)"'
alias gcp-project='gcloud config get-value project'
alias gcp-projects='gcloud projects list'
```

## Quick Troubleshooting

```bash
# Check authentication
gcloud auth list

# Check current configuration
gcloud config list

# Test API access
gcloud services list --enabled

# Check quotas
gcloud compute project-info describe --project=$(gcloud config get-value project)

# View recent operations
gcloud logging read "protoPayload.methodName!=null" --limit 20
```