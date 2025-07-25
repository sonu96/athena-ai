# GCP & CDP Setup Guide for New Projects

This guide documents how to reuse the Google Cloud Platform (GCP) and Coinbase Developer Platform (CDP) setup from the Athena AI project in new projects.

## Table of Contents
1. [Core Components to Copy](#core-components-to-copy)
2. [GCP Setup](#gcp-setup)
3. [CDP Setup](#cdp-setup)
4. [Environment Configuration](#environment-configuration)
5. [Quick Start for New Project](#quick-start-for-new-project)

## Core Components to Copy

### Essential Files and Directories

```bash
# Core CDP integration
src/cdp/
├── __init__.py
├── base_client.py          # CDP wallet and transaction management
└── version_check.py        # CDP SDK version verification

# Blockchain utilities
src/blockchain/
├── __init__.py
└── rpc_reader.py          # Direct RPC calls without web3

# GCP integrations
src/gcp/
├── __init__.py
├── firestore_client.py    # Firestore database wrapper
├── pubsub_client.py       # Pub/Sub messaging
└── secret_manager.py      # Secret Manager integration

# Configuration
config/
├── __init__.py
├── settings.py            # Pydantic settings with Secret Manager
└── contracts.py           # Aerodrome contract addresses and ABIs

# Scripts
scripts/
├── setup_secrets.py       # Initialize GCP secrets
└── update_cdp_config.py   # Update CDP credentials

# Environment and deployment
.env.example               # Environment template
requirements.txt           # Core dependencies
Dockerfile                # Production container
deployment/               # Cloud Run configs
```

### Key Dependencies (requirements.txt)

```txt
# Core dependencies to include
cdp-sdk>=1.23.0
google-cloud-secret-manager>=2.16.0
google-cloud-firestore>=2.11.0
google-cloud-pubsub>=2.18.0
google-cloud-logging>=3.5.0
google-generativeai>=0.3.0
pydantic>=2.11.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
```

## GCP Setup

### 1. Create GCP Project

```bash
# Create new project
gcloud projects create YOUR-PROJECT-ID --name="Your Project Name"

# Set as active project
gcloud config set project YOUR-PROJECT-ID

# Enable required APIs
gcloud services enable \
    secretmanager.googleapis.com \
    firestore.googleapis.com \
    pubsub.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com
```

### 2. Setup Service Account

```bash
# Create service account
gcloud iam service-accounts create athena-sa \
    --display-name="Athena Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:athena-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:athena-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/datastore.user"

gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:athena-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.editor"
```

### 3. Initialize Firestore

```bash
# Create Firestore database
gcloud firestore databases create --location=us-central1
```

## CDP Setup

### 1. Get CDP Credentials

1. Sign up at [CDP Portal](https://portal.cdp.coinbase.com/)
2. Create an API key with required scopes:
   - `wallet:create`
   - `wallet:read`
   - `wallet:transactions:send`
3. Save the API key ID and secret

### 2. Get CDP Client API Key (for RPC)

1. In CDP Portal, create a Client API key
2. This provides authenticated RPC access to Base network
3. Save the client API key

### 3. Store Credentials in Secret Manager

```bash
# Store CDP API credentials
echo -n "your-cdp-api-key-id" | gcloud secrets create cdp-api-key --data-file=-
echo -n "your-cdp-api-key-secret" | gcloud secrets create cdp-api-secret --data-file=-
echo -n "your-cdp-client-api-key" | gcloud secrets create cdp-client-api-key --data-file=-

# Generate and store wallet secret (32-byte hex)
python -c "import secrets; print(secrets.token_hex(32))" | gcloud secrets create cdp-wallet-secret --data-file=-
```

## Environment Configuration

### .env File Structure

```bash
# Google Cloud Configuration (REQUIRED)
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
FIRESTORE_DATABASE=(default)

# Google AI Configuration
GOOGLE_AI_MODEL=gemini-1.5-flash
GOOGLE_LOCATION=us-central1

# CDP Configuration (Optional - will use Secret Manager if not set)
# CDP_API_KEY_ID=your_cdp_api_key_id_here
# CDP_API_KEY_SECRET=your_cdp_api_key_secret_here
# CDP_CLIENT_API_KEY=your_cdp_client_api_key_here

# Base Chain Configuration
BASE_RPC_URL=https://mainnet.base.org
CHAIN_ID=8453

# Agent Configuration
AGENT_WALLET_ID=  # Will be created on first run
```

## Quick Start for New Project

### 1. Copy Core Files

```bash
# Create new project directory
mkdir my-new-project
cd my-new-project

# Copy essential directories from athena-ai
cp -r /path/to/athena-ai/src/cdp src/
cp -r /path/to/athena-ai/src/blockchain src/
cp -r /path/to/athena-ai/src/gcp src/
cp -r /path/to/athena-ai/config .
cp -r /path/to/athena-ai/scripts .

# Copy configuration files
cp /path/to/athena-ai/.env.example .
cp /path/to/athena-ai/requirements.txt .
cp /path/to/athena-ai/Dockerfile .
```

### 2. Initialize New Project

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your GCP_PROJECT_ID

# Initialize GCP authentication
gcloud auth application-default login
```

### 3. Test CDP Integration

```python
# test_cdp.py
import asyncio
from src.cdp.base_client import BaseClient

async def test():
    client = BaseClient()
    await client.initialize()
    print(f"Wallet: {client.wallet.addresses[0].address}")
    
    # Test getting balance
    balance = await client.get_balance("USDC")
    print(f"USDC Balance: {balance}")

asyncio.run(test())
```

### 4. Test GCP Integration

```python
# test_gcp.py
from src.gcp.firestore_client import FirestoreClient
from src.gcp.secret_manager import get_secret

# Test Firestore
firestore = FirestoreClient()
await firestore.initialize()

# Test Secret Manager
secret = get_secret("cdp-api-key")
print("Secret loaded successfully" if secret else "Failed to load secret")
```

## Key Features Provided

### CDP Base Client
- Automatic wallet creation/loading
- Transaction building and signing
- Token balance queries
- Pool information retrieval
- Gas estimation

### RPC Reader
- Direct blockchain queries without web3
- Contract function calls
- Event log fetching
- Storage slot reading

### GCP Integration
- Automatic secret loading with fallback to env vars
- Firestore async operations
- Pub/Sub message handling
- Structured logging

### Settings Management
- Pydantic validation
- Secret Manager integration
- Environment variable fallback
- Type-safe configuration

## Deployment

### Cloud Run Deployment

```bash
# Deploy to Cloud Run
gcloud run deploy my-service \
    --source . \
    --region us-central1 \
    --service-account athena-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com \
    --set-env-vars GCP_PROJECT_ID=YOUR-PROJECT-ID
```

## Common Issues and Solutions

### 1. CDP SDK Installation
If you encounter Rust compilation errors:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Secret Manager Permissions
Ensure your service account has the `secretmanager.secretAccessor` role.

### 3. CDP Wallet Persistence
The wallet ID is automatically saved to Firestore. On restart, it will load the existing wallet.

## Security Best Practices

1. **Never commit secrets** - Use Secret Manager
2. **Rotate CDP credentials** regularly using `scripts/update_cdp_config.py`
3. **Use service accounts** for production deployments
4. **Enable audit logging** for Secret Manager access
5. **Implement transaction limits** in your application logic

## Additional Resources

- [CDP Documentation](https://docs.cdp.coinbase.com/)
- [GCP Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firestore Best Practices](https://cloud.google.com/firestore/docs/best-practices)