#!/bin/bash

# Athena AI - Google Cloud Project Setup Script
# This script sets up a new GCP project and cleans up old ones

set -e  # Exit on error

echo "🚀 Athena AI - Google Cloud Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get current account
CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null)
echo -e "\nCurrent account: ${GREEN}${CURRENT_ACCOUNT}${NC}"

# List existing projects
echo -e "\n📋 Current Google Cloud Projects:"
echo "--------------------------------"
gcloud projects list --format="table(projectId,name,createTime)"

# Ask if user wants to delete old projects
echo -e "\n🗑️  Project Cleanup"
read -p "Do you want to delete any existing projects? (y/N): " DELETE_PROJECTS

if [[ $DELETE_PROJECTS =~ ^[Yy]$ ]]; then
    echo "Enter project IDs to delete (comma-separated, or 'skip' to continue):"
    read -p "> " PROJECTS_TO_DELETE
    
    if [ "$PROJECTS_TO_DELETE" != "skip" ] && [ -n "$PROJECTS_TO_DELETE" ]; then
        IFS=',' read -ra PROJECT_ARRAY <<< "$PROJECTS_TO_DELETE"
        for project in "${PROJECT_ARRAY[@]}"; do
            project=$(echo "$project" | xargs)  # Trim whitespace
            echo -e "\nDeleting project: $project"
            if gcloud projects delete "$project" --quiet; then
                print_status "Deleted project: $project"
            else
                print_error "Failed to delete project: $project"
            fi
        done
    fi
fi

# Create new project for Athena
echo -e "\n🆕 Create New Project for Athena AI"
echo "-----------------------------------"

# Generate suggested project ID
SUGGESTED_ID="athena-ai-$(date +%Y%m%d)"
echo "Suggested project ID: $SUGGESTED_ID"

read -p "Enter project ID (or press Enter for suggested): " PROJECT_ID
PROJECT_ID=${PROJECT_ID:-$SUGGESTED_ID}

read -p "Enter project name (default: Athena AI): " PROJECT_NAME
PROJECT_NAME=${PROJECT_NAME:-"Athena AI"}

# Check if project ID is available
if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
    print_error "Project ID '$PROJECT_ID' already exists!"
    exit 1
fi

# Create project
echo -e "\nCreating project..."
if gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME"; then
    print_status "Created project: $PROJECT_ID"
else
    print_error "Failed to create project"
    exit 1
fi

# Set as active project
gcloud config set project "$PROJECT_ID"
print_status "Set $PROJECT_ID as active project"

# Get billing account
echo -e "\n💳 Billing Setup"
echo "----------------"
BILLING_ACCOUNTS=$(gcloud billing accounts list --format="value(name)")

if [ -z "$BILLING_ACCOUNTS" ]; then
    print_warning "No billing accounts found. Please set up billing in the console."
else
    echo "Available billing accounts:"
    gcloud billing accounts list --format="table(displayName,name)"
    
    if [ $(echo "$BILLING_ACCOUNTS" | wc -l) -eq 1 ]; then
        # Only one billing account, use it
        gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNTS"
        print_status "Linked billing account"
    else
        echo "Multiple billing accounts found. Please link manually:"
        echo "gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID"
    fi
fi

# Enable required APIs
echo -e "\n🔧 Enabling Required APIs"
echo "------------------------"

APIS=(
    "secretmanager.googleapis.com"
    "aiplatform.googleapis.com"
    "firestore.googleapis.com"
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "pubsub.googleapis.com"
    "cloudscheduler.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo -n "Enabling $api... "
    if gcloud services enable "$api" --quiet; then
        print_status "enabled"
    else
        print_error "failed"
    fi
done

# Create Firestore database
echo -e "\n🗄️  Setting up Firestore"
echo "----------------------"
if gcloud firestore databases describe --quiet &>/dev/null; then
    print_status "Firestore database already exists"
else
    echo "Creating Firestore database..."
    if gcloud firestore databases create --location=us-central1 --quiet; then
        print_status "Created Firestore database"
    else
        print_warning "Failed to create Firestore database (may already exist)"
    fi
fi

# Set up application default credentials
echo -e "\n🔐 Setting up Authentication"
echo "---------------------------"
gcloud auth application-default login
print_status "Set up application default credentials"

# Create .env file
echo -e "\n📝 Creating .env file"
echo "--------------------"
cat > .env << EOF
# Google Cloud Configuration (REQUIRED)
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=us-central1
FIRESTORE_DATABASE=(default)

# Google AI Configuration
GOOGLE_AI_MODEL=gemini-1.5-pro
GOOGLE_LOCATION=us-central1

# Base Chain Configuration
BASE_RPC_URL=https://mainnet.base.org
CHAIN_ID=8453

# Agent Configuration
AGENT_CYCLE_TIME=300
AGENT_MAX_POSITION_SIZE=1000
AGENT_RISK_LIMIT=0.02
LOG_LEVEL=INFO

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
ENABLE_CORS=true

# LangSmith Configuration
LANGSMITH_PROJECT=athena-ai

# Monitoring
ENABLE_MONITORING=true
EOF

print_status "Created .env file with project configuration"

# Summary
echo -e "\n✅ Setup Complete!"
echo "=================="
echo -e "Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "Project Name: ${GREEN}$PROJECT_NAME${NC}"
echo -e "\nNext steps:"
echo "1. Run: python scripts/setup_secrets.py"
echo "2. Run: python run.py"
echo -e "\nUseful commands:"
echo "- View project: gcloud projects describe $PROJECT_ID"
echo "- View logs: gcloud logging read 'resource.type=cloud_run_revision'"
echo "- Deploy: gcloud run deploy athena-ai --source ."

# Export for use in other scripts
export GCP_PROJECT_ID=$PROJECT_ID

echo -e "\n🎉 Happy trading with Athena AI!"