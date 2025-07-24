#!/bin/bash
# Monitoring script for Athena AI

echo "🔍 Athena AI Monitoring Dashboard"
echo "================================="

# Service status
echo -e "\n📊 Service Status:"
gcloud run services describe athena-ai --region us-central1 --format="value(status.url)" 2>/dev/null && echo "✅ Service is deployed" || echo "❌ Service not deployed"

# Recent logs
echo -e "\n📝 Recent Logs (last 10 entries):"
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai" --limit 10 --format "table(timestamp,severity,textPayload)"

# Errors
echo -e "\n❌ Recent Errors:"
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai AND severity>=ERROR" --limit 5 --format "table(timestamp,textPayload)"

# Build status
echo -e "\n🏗️ Recent Builds:"
gcloud builds list --limit 3 --format="table(ID,STATUS,CREATE_TIME)"

# Cloud Run revisions
echo -e "\n🔄 Recent Revisions:"
gcloud run revisions list --service athena-ai --region us-central1 --limit 3 --format="table(REVISION,ACTIVE,CREATED)"

echo -e "\n💡 Useful commands:"
echo "- View live logs: gcloud logging tail \"resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai\""
echo "- View build logs: gcloud builds log <BUILD_ID>"
echo "- Check service URL: gcloud run services describe athena-ai --region us-central1 --format=\"value(status.url)\""