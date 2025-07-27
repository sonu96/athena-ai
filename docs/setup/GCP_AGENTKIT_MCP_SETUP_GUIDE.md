# GCP, AgentKit & MCP Setup Guide

This guide documents how to set up Google Cloud Platform (GCP), Coinbase AgentKit, and QuickNode MCP for the Athena AI project.

## Table of Contents
1. [Overview](#overview)
2. [GCP Setup](#gcp-setup)
3. [AgentKit Setup](#agentkit-setup)
4. [QuickNode MCP Setup](#quicknode-mcp-setup)
5. [Environment Configuration](#environment-configuration)
6. [Quick Start](#quick-start)

## Overview

Athena AI uses a streamlined architecture with three core components:

- **Google Cloud Platform (GCP)**: Infrastructure, storage, and secret management
- **Coinbase AgentKit**: AI-native blockchain transactions using existing CDP credentials
- **QuickNode MCP**: Natural language blockchain data queries and analytics

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
```

### 3. Initialize Firestore

```bash
# Create Firestore database
gcloud firestore databases create --location=us-central1
```

## AgentKit Setup

AgentKit uses your existing CDP (Coinbase Developer Platform) credentials, so no new registration is needed.

### 1. Verify CDP Credentials

If you already have CDP credentials from the previous setup:
- CDP API Key ID
- CDP API Secret
- These will be reused by AgentKit

If you need new credentials:
1. Sign up at [CDP Portal](https://portal.cdp.coinbase.com/)
2. Create an API key with required scopes:
   - `wallet:create`
   - `wallet:read`
   - `wallet:transactions:send`

### 2. Store Credentials in Secret Manager

```bash
# Store CDP API credentials (same as before)
echo -n "your-cdp-api-key-id" | gcloud secrets create cdp-api-key --data-file=-
echo -n "your-cdp-api-key-secret" | gcloud secrets create cdp-api-secret --data-file=-

# Wallet secret is auto-generated if not provided
```

## QuickNode MCP Setup

### 1. Get QuickNode API Key

1. Sign up at [QuickNode](https://www.quicknode.com/)
2. Create an endpoint for Base network
3. Add the Aerodrome addon (if available)
4. Get your API key

### 2. Store QuickNode Credentials

```bash
# Store QuickNode API key
echo -n "your-quicknode-api-key" | gcloud secrets create quicknode-api-key --data-file=-
```

### 3. Install MCP Server

```bash
# Install globally
npm install -g @quicknode/mcp

# Or use npx (no installation needed)
npx @quicknode/mcp
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

# CDP Configuration (AgentKit will use these)
# CDP_API_KEY_ID=your_cdp_api_key_id_here
# CDP_API_KEY_SECRET=your_cdp_api_key_secret_here

# QuickNode Configuration
# QUICKNODE_API_KEY=your_quicknode_api_key_here

# Agent Configuration
AGENT_WALLET_ID=  # Will be created on first run or reuse existing
AGENT_CYCLE_TIME=300  # 5 minutes between cycles
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-repo/athena-ai.git
cd athena-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies for MCP
npm install
```

### 2. Configure Environment

```bash
# Copy example environment
cp .env.example .env

# Edit with your values
nano .env
```

### 3. Initialize GCP Authentication

```bash
gcloud auth application-default login
```

### 4. Test Components

```python
# test_setup.py
import asyncio
from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agentkit.agent_client import AthenaAgentKit

async def test():
    # Test MCP
    mcp = QuickNodeMCP(settings.quicknode_api_key)
    await mcp.initialize()
    pools = await mcp.get_aerodrome_pools(min_apr=20)
    print(f"MCP Test: Found {len(pools)} pools")
    
    # Test AgentKit
    agentkit = AthenaAgentKit()
    await agentkit.initialize()
    print(f"AgentKit Test: Wallet {agentkit.address}")
    
    # Cleanup
    await mcp.close()

asyncio.run(test())
```

### 5. Run Athena

```bash
# Run with API server
python run_new.py

# Or run standalone
python main_new.py
```

## Key Differences from CDP Setup

### Simplified Architecture

**Before (CDP SDK):**
- 700+ lines of custom blockchain code
- Complex RPC management
- Manual error handling
- Custom transaction building

**Now (AgentKit + MCP):**
- ~100 lines of integration code
- Natural language queries
- Automatic error handling
- AI-native operations

### Example Code Comparison

**Before:**
```python
# Complex pool query
pool_info = await base_client.get_pool_info("WETH", "USDC")
reserves = await rpc_reader.get_reserves(pool_address)
apr = await calculate_apr(emissions, tvl, price)
```

**After:**
```python
# Simple natural language query
pool_data = await mcp.query("Get WETH/USDC pool info with APR")
```

## Deployment

### Cloud Run Deployment

```bash
# Deploy to Cloud Run
gcloud run deploy athena-ai \
    --source . \
    --region us-central1 \
    --service-account athena-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com \
    --set-env-vars GCP_PROJECT_ID=YOUR-PROJECT-ID
```

## Troubleshooting

### MCP Connection Issues

```bash
# Check MCP server is accessible
npx @quicknode/mcp --version

# Test with curl
curl -X POST https://api.quicknode.com/v1/test
```

### AgentKit Initialization

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **"CDP credentials not found"**: Ensure secrets are in Secret Manager
2. **"MCP server not responding"**: Check QuickNode API key is valid
3. **"Wallet creation failed"**: Verify CDP API key has correct permissions

## Security Best Practices

1. **Never commit API keys** - Always use Secret Manager
2. **Rotate credentials regularly**
3. **Use service accounts** for production
4. **Enable audit logging** for all secret access
5. **Set transaction limits** in agent configuration

## Next Steps

1. Monitor agent performance in Cloud Logging
2. Set up alerts for anomalies
3. Configure backup strategies
4. Implement additional safety checks

## Resources

- [Coinbase AgentKit Docs](https://docs.cdp.coinbase.com/agentkit)
- [QuickNode MCP Docs](https://www.quicknode.com/docs/mcp)
- [GCP Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Athena AI Architecture](../ARCHITECTURE.md)