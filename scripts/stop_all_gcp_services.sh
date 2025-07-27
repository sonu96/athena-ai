#!/bin/bash

# Script to stop all GCP services and delete data for the Athena project
# This will help avoid any charges

PROJECT_ID="athena-defi-agent-1752635199"

echo "🛑 Stopping all GCP services for project: $PROJECT_ID"
echo "⚠️  This will delete all data and stop all services to avoid charges"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Set the project
echo "Setting project..."
gcloud config set project $PROJECT_ID

# 1. Stop Cloud Run services
echo ""
echo "1️⃣ Stopping Cloud Run services..."
gcloud run services list --region=us-central1 --format="value(metadata.name)" | while read service; do
    echo "   Deleting Cloud Run service: $service"
    gcloud run services delete $service --region=us-central1 --quiet
done

# 2. Delete Firestore data
echo ""
echo "2️⃣ Deleting Firestore collections..."
# List of collections to delete
collections=(
    "agent_state"
    "agent_cycles" 
    "performance_metrics"
    "observed_patterns"
    "observation_metrics"
    "pattern_confidence"
    "pool_profiles"
    "pool_metrics"
    "pattern_correlations"
    "pool_observations"
    "market_theories"
    "trading_decisions"
    "gas_patterns"
)

for collection in "${collections[@]}"; do
    echo "   Deleting collection: $collection"
    gcloud firestore delete $collection --recursive --quiet 2>/dev/null || echo "   Collection $collection not found or already deleted"
done

# 3. Delete Pub/Sub topics and subscriptions
echo ""
echo "3️⃣ Deleting Pub/Sub resources..."
# Delete subscriptions first
gcloud pubsub subscriptions list --format="value(name)" | while read sub; do
    echo "   Deleting subscription: $sub"
    gcloud pubsub subscriptions delete $sub --quiet
done

# Then delete topics
gcloud pubsub topics list --format="value(name)" | while read topic; do
    echo "   Deleting topic: $topic"
    gcloud pubsub topics delete $topic --quiet
done

# 4. Delete Cloud Scheduler jobs
echo ""
echo "4️⃣ Deleting Cloud Scheduler jobs..."
gcloud scheduler jobs list --location=us-central1 --format="value(name)" | while read job; do
    echo "   Deleting scheduler job: $job"
    gcloud scheduler jobs delete $job --location=us-central1 --quiet
done

# 5. Delete Cloud Functions
echo ""
echo "5️⃣ Deleting Cloud Functions..."
gcloud functions list --format="value(name)" | while read func; do
    echo "   Deleting function: $func"
    gcloud functions delete $func --region=us-central1 --quiet
done

# 6. Delete BigQuery datasets
echo ""
echo "6️⃣ Deleting BigQuery datasets..."
datasets=(
    "aerodrome_data"
    "agent_metrics"
)

for dataset in "${datasets[@]}"; do
    echo "   Deleting dataset: $dataset"
    bq rm -r -f -d $PROJECT_ID:$dataset 2>/dev/null || echo "   Dataset $dataset not found"
done

# 7. Keep Secret Manager (secrets are preserved)
echo ""
echo "7️⃣ Keeping Secret Manager secrets..."
echo "   ✅ Secrets are preserved and will not be deleted"
gcloud secrets list --format="table(name,createTime)"

# 8. Delete any Cloud Storage buckets
echo ""
echo "8️⃣ Checking Cloud Storage buckets..."
gsutil ls 2>/dev/null | grep "gs://" | while read bucket; do
    if [[ $bucket == *"$PROJECT_ID"* ]]; then
        echo "   Found bucket: $bucket"
        read -p "   Delete this bucket and all its contents? (yes/no): " del_bucket
        if [ "$del_bucket" == "yes" ]; then
            gsutil -m rm -r $bucket
        fi
    fi
done

# 9. Check for any Container Registry images
echo ""
echo "9️⃣ Checking Container Registry..."
gcloud container images list --format="value(name)" | while read image; do
    echo "   Found image: $image"
    echo "   To delete: gcloud container images delete $image --quiet"
done

# 10. Budget alerts
echo ""
echo "🔟 Budget alerts status:"
gcloud billing budgets list --billing-account=$(gcloud beta billing projects describe $PROJECT_ID --format="value(billingAccountName)") 2>/dev/null || echo "   Could not check budget alerts"

echo ""
echo "✅ Service shutdown complete!"
echo ""
echo "📊 Current status:"
echo "   - Cloud Run services: Deleted"
echo "   - Firestore data: Deleted"
echo "   - Pub/Sub resources: Deleted"
echo "   - Cloud Scheduler: Deleted"
echo "   - Cloud Functions: Deleted"
echo "   - BigQuery datasets: Deleted"
echo ""
echo "💰 To minimize costs further:"
echo "   1. Delete any remaining secrets manually if not needed"
echo "   2. Delete the project entirely if no longer needed:"
echo "      gcloud projects delete $PROJECT_ID"
echo ""
echo "📈 Monitor billing at: https://console.cloud.google.com/billing"