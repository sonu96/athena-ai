#!/bin/bash

# Script to recreate all GCP services for the Athena project
# This assumes Secret Manager is already set up with necessary secrets

PROJECT_ID="athena-defi-agent-1752635199"
REGION="us-central1"
SERVICE_NAME="athena-ai"

echo "🚀 Recreating GCP services for project: $PROJECT_ID"
echo "✅ This assumes Secret Manager already contains all necessary secrets"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Set the project
echo "Setting project..."
gcloud config set project $PROJECT_ID

# 1. Enable necessary APIs
echo ""
echo "1️⃣ Enabling required APIs..."
apis=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "firestore.googleapis.com"
    "pubsub.googleapis.com"
    "cloudscheduler.googleapis.com"
    "cloudfunctions.googleapis.com"
    "secretmanager.googleapis.com"
    "bigquery.googleapis.com"
    "artifactregistry.googleapis.com"
)

for api in "${apis[@]}"; do
    echo "   Enabling $api..."
    gcloud services enable $api
done

# 2. Create Firestore database (if not exists)
echo ""
echo "2️⃣ Setting up Firestore..."
gcloud firestore databases create --location=$REGION --type=firestore-native 2>/dev/null || echo "   Firestore database already exists"

# Create Firestore indexes
echo "   Creating Firestore indexes..."
cat > firestore.indexes.json << 'EOF'
{
  "indexes": [
    {
      "collectionGroup": "agent_cycles",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "timestamp", "order": "DESCENDING" }
      ]
    },
    {
      "collectionGroup": "pool_profiles",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "pool_address", "order": "ASCENDING" },
        { "fieldPath": "last_updated", "order": "DESCENDING" }
      ]
    }
  ]
}
EOF

gcloud firestore indexes create --file=firestore.indexes.json 2>/dev/null || echo "   Some indexes may already exist"
rm firestore.indexes.json

# 3. Create Pub/Sub topics
echo ""
echo "3️⃣ Creating Pub/Sub topics..."
topics=(
    "pool-observations"
    "market-events"
    "agent-decisions"
    "gas-updates"
)

for topic in "${topics[@]}"; do
    echo "   Creating topic: $topic"
    gcloud pubsub topics create $topic 2>/dev/null || echo "   Topic $topic already exists"
done

# 4. Create BigQuery datasets
echo ""
echo "4️⃣ Creating BigQuery datasets..."
datasets=(
    "aerodrome_data"
    "agent_metrics"
)

for dataset in "${datasets[@]}"; do
    echo "   Creating dataset: $dataset"
    bq mk -d --location=$REGION --description="Athena AI $dataset" $PROJECT_ID:$dataset 2>/dev/null || echo "   Dataset $dataset already exists"
done

# Create BigQuery tables
echo "   Creating BigQuery tables..."

# Pool observations table
bq mk -t \
  --description="Pool observations from Aerodrome" \
  $PROJECT_ID:aerodrome_data.pool_observations \
  timestamp:TIMESTAMP,pool_address:STRING,tvl:FLOAT64,apr:FLOAT64,volume_24h:FLOAT64,fee_apr:FLOAT64,emission_apr:FLOAT64 2>/dev/null || echo "   Table pool_observations already exists"

# Agent metrics table
bq mk -t \
  --description="Agent performance metrics" \
  $PROJECT_ID:agent_metrics.performance \
  timestamp:TIMESTAMP,cycle_number:INTEGER,profit:FLOAT64,gas_spent:FLOAT64,trades_executed:INTEGER,success_rate:FLOAT64 2>/dev/null || echo "   Table performance already exists"

# 5. Build and deploy Cloud Run service
echo ""
echo "5️⃣ Building and deploying Cloud Run service..."

# Check if Dockerfile exists
if [ ! -f "athena-ai/Dockerfile" ]; then
    echo "   ❌ Dockerfile not found at athena-ai/Dockerfile"
    echo "   Skipping Cloud Run deployment"
else
    # Build using Cloud Build
    echo "   Building container image..."
    gcloud builds submit athena-ai \
        --tag gcr.io/$PROJECT_ID/$SERVICE_NAME \
        --timeout=30m

    # Deploy to Cloud Run
    echo "   Deploying to Cloud Run..."
    gcloud run deploy $SERVICE_NAME \
        --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
        --platform managed \
        --region $REGION \
        --memory 4Gi \
        --cpu 2 \
        --timeout 900 \
        --max-instances 5 \
        --min-instances 1 \
        --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION" \
        --service-account="$SERVICE_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --allow-unauthenticated
fi

# 6. Create Cloud Scheduler jobs
echo ""
echo "6️⃣ Creating Cloud Scheduler jobs..."

# Get Cloud Run service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" 2>/dev/null)

if [ ! -z "$SERVICE_URL" ]; then
    # Create scheduler job for agent cycles
    echo "   Creating agent cycle scheduler..."
    gcloud scheduler jobs create http agent-cycle-trigger \
        --location=$REGION \
        --schedule="*/5 * * * *" \
        --uri="$SERVICE_URL/cycle" \
        --http-method=POST \
        --attempt-deadline=900s \
        2>/dev/null || echo "   Scheduler job already exists"
else
    echo "   ⚠️  Cloud Run service URL not found, skipping scheduler creation"
fi

# 7. Set up monitoring and alerts
echo ""
echo "7️⃣ Setting up monitoring..."

# Create budget alert
BILLING_ACCOUNT=$(gcloud billing projects describe $PROJECT_ID --format="value(billingAccountName)" 2>/dev/null)
if [ ! -z "$BILLING_ACCOUNT" ]; then
    echo "   Creating budget alert..."
    gcloud billing budgets create \
        --billing-account=$BILLING_ACCOUNT \
        --display-name="Athena AI Budget Alert" \
        --budget-amount=30 \
        --threshold-rule=percent=50 \
        --threshold-rule=percent=90 \
        --threshold-rule=percent=100 \
        2>/dev/null || echo "   Budget alert may already exist"
fi

# 8. Create service account (if needed)
echo ""
echo "8️⃣ Setting up service account..."
SERVICE_ACCOUNT="$SERVICE_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gcloud iam service-accounts create $SERVICE_NAME \
    --display-name="Athena AI Service Account" \
    2>/dev/null || echo "   Service account already exists"

# Grant necessary permissions
echo "   Granting permissions..."
roles=(
    "roles/firestore.serviceAgent"
    "roles/secretmanager.secretAccessor"
    "roles/pubsub.publisher"
    "roles/pubsub.subscriber"
    "roles/bigquery.dataEditor"
    "roles/logging.logWriter"
)

for role in "${roles[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$role" \
        --condition=None \
        2>/dev/null || echo "   Role $role may already be assigned"
done

echo ""
echo "✅ Service recreation complete!"
echo ""
echo "📊 Created resources:"
echo "   - APIs: Enabled"
echo "   - Firestore: Database and indexes created"
echo "   - Pub/Sub: Topics created"
echo "   - BigQuery: Datasets and tables created"
echo "   - Cloud Run: Service deployed (if Dockerfile exists)"
echo "   - Cloud Scheduler: Jobs created"
echo "   - Service Account: Created with permissions"
echo ""
echo "🔑 Secrets are preserved in Secret Manager"
echo ""
echo "📈 Next steps:"
echo "   1. Verify all services are running: gcloud run services list"
echo "   2. Check logs: gcloud logging read --limit 50"
echo "   3. Monitor costs: https://console.cloud.google.com/billing"
echo ""
echo "🚀 To start the agent locally:"
echo "   cd athena-ai"
echo "   source venv/bin/activate"
echo "   python run.py"