# GCP Services Shutdown Guide

## Quick Shutdown

Run the shutdown script:
```bash
./scripts/stop_all_gcp_services.sh
```

This script will:
1. **Delete all Cloud Run services** - No more compute charges
2. **Delete all Firestore data** - No more storage charges
3. **Delete Pub/Sub topics and subscriptions** - No more messaging charges
4. **Delete Cloud Scheduler jobs** - No more scheduled tasks
5. **Delete Cloud Functions** - No more serverless compute
6. **Delete BigQuery datasets** - No more analytics storage
7. **List secrets** (manual deletion required)
8. **Check for Cloud Storage buckets** - Delete if found
9. **Check Container Registry** - List images for manual deletion

## Manual Steps Required

### 1. Delete Secrets (if not needed)
```bash
# List all secrets
gcloud secrets list

# Delete a specific secret
gcloud secrets delete SECRET_NAME --quiet
```

### 2. Check for Any Remaining Resources
```bash
# Check all services
gcloud services list --enabled

# Check for any compute instances
gcloud compute instances list

# Check for load balancers
gcloud compute forwarding-rules list

# Check for persistent disks
gcloud compute disks list
```

### 3. Disable Expensive APIs
```bash
# Disable APIs that might incur charges
gcloud services disable secretmanager.googleapis.com
gcloud services disable run.googleapis.com
gcloud services disable cloudfunctions.googleapis.com
gcloud services disable cloudscheduler.googleapis.com
gcloud services disable pubsub.googleapis.com
gcloud services disable firestore.googleapis.com
gcloud services disable bigquery.googleapis.com
```

### 4. Delete the Entire Project (Nuclear Option)
If you want to ensure absolutely no charges:
```bash
gcloud projects delete athena-defi-agent-1752635199
```

## Cost Monitoring

1. **Check current billing**: https://console.cloud.google.com/billing
2. **Set up budget alerts** if keeping the project
3. **Export billing data** for analysis

## What Gets Deleted

- ✅ All Cloud Run services (athena-ai)
- ✅ All Firestore collections and documents
- ✅ All Pub/Sub topics and subscriptions  
- ✅ All Cloud Scheduler jobs
- ✅ All Cloud Functions
- ✅ All BigQuery datasets and tables
- ❌ Secrets (KEPT - not deleted)
- ❓ Container images (manual deletion required)
- ❓ Cloud Storage buckets (prompted for deletion)

## What Is Kept

- ✅ **Secret Manager**: All secrets are preserved
- ✅ **Service Accounts**: Not deleted (may be needed for recreation)
- ✅ **APIs**: Remain enabled (can be disabled manually)

## Verification Commands

After running the shutdown script, verify everything is deleted:

```bash
# Verify no Cloud Run services
gcloud run services list --region=us-central1

# Verify no Firestore data (check in console)
# https://console.cloud.google.com/firestore

# Verify no active resources
gcloud asset search-all-resources --scope=projects/athena-defi-agent-1752635199
```

## Free Tier Considerations

Some services have free tiers that won't incur charges:
- Firestore: 1GB storage free
- Cloud Run: 2 million requests free/month
- Pub/Sub: 10GB free/month
- Secret Manager: 6 active secrets free

However, it's better to delete everything if you're not using the project.

## Recreating Services

To recreate all services after shutdown:
```bash
./scripts/recreate_gcp_services.sh
```

This will:
1. Enable all necessary APIs
2. Create Firestore database and indexes
3. Create Pub/Sub topics
4. Create BigQuery datasets and tables
5. Build and deploy Cloud Run service
6. Set up Cloud Scheduler jobs
7. Configure monitoring and budget alerts
8. Create service accounts with proper permissions

**Note**: Since secrets are preserved, the recreation will work seamlessly.

## Important Notes

- **Secrets**: Kept intact during shutdown for easy recreation
- **Project Deletion**: Deleting the project is permanent and cannot be undone
- **Billing**: Charges may appear up to 24 hours after resource deletion
- **APIs**: Disabling APIs doesn't delete resources, delete resources first
- **Recreation**: Can quickly spin up all services again using the recreate script