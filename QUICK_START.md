# 🚀 Athena AI - Quick Start Guide

This guide will get you up and running with Athena AI in minutes.

## Prerequisites

- Google Cloud account
- `gcloud` CLI installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
- Python 3.11+
- CDP account from Coinbase ([Get API Keys](https://portal.cdp.coinbase.com/))

## Step 1: Set Up Google Cloud Project

Run the automated setup script:

```bash
./scripts/gcp_setup.sh
```

This script will:
- List your existing projects and optionally delete old ones
- Create a new project for Athena AI
- Enable all required Google Cloud APIs
- Set up Firestore database
- Configure authentication
- Create a `.env` file with your project settings

## Step 2: Configure Secrets

After the project is created, set up your secrets:

```bash
python scripts/setup_secrets.py
```

You'll need:
- **CDP API Key** (required): From Coinbase Developer Portal
- **CDP API Secret** (required): From Coinbase Developer Portal
- **LangSmith API Key** (optional): For observability
- **Mem0 API Key** (optional): For cloud memory storage

## Step 3: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Step 4: Run Athena

```bash
python run.py
```

You should see:
```
🚀 Initializing Athena AI with API...
🧠 I am learning 24/7 to maximize DeFi profits on Aerodrome
💳 Wallet address: 0x...
🌐 API server starting on http://0.0.0.0:8000
📚 API docs available at http://0.0.0.0:8000/docs
👀 Starting 24/7 monitoring...
📊 Tracking gas prices, pool APRs, and market opportunities
```

## Step 5: Monitor Performance

Open your browser to:
- **API Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Common Commands

### Check agent performance
```bash
curl http://localhost:8000/performance/24h
```

### View current positions
```bash
curl http://localhost:8000/positions
```

### Get gas recommendations
```bash
curl http://localhost:8000/gas/recommendation
```

### Watch live updates (WebSocket)
```javascript
const ws = new WebSocket('ws://localhost:8000/live');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

## Project Management

### Clean up old projects
```bash
./scripts/gcp_cleanup.sh
```

### Deploy to production
```bash
gcloud run deploy athena-ai --source . --region us-central1
```

### View logs
```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

## Troubleshooting

### "Permission denied" errors
```bash
# Re-authenticate
gcloud auth application-default login
```

### "Project not set" errors
```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### "API not enabled" errors
```bash
# Enable the API
gcloud services enable SERVICE_NAME.googleapis.com
```

### Secret Manager errors
```bash
# Check if secrets exist
gcloud secrets list

# View secret versions
gcloud secrets versions list SECRET_NAME
```

## Next Steps

1. **Fund your wallet**: Send some ETH to the wallet address shown on startup
2. **Monitor performance**: Check the API endpoints regularly
3. **Review memories**: See what patterns Athena discovers at `/memories/recent`
4. **Adjust settings**: Modify risk limits and strategies in `config/settings.py`

## Important Notes

- Athena starts with a cautious approach and becomes more confident over time
- The first 24 hours are primarily for learning and observation
- Keep the agent running continuously for best results
- Monitor logs for any errors or important discoveries

## Support

- Check logs: `tail -f logs/athena.log`
- View API docs: http://localhost:8000/docs
- Debug mode: Set `LOG_LEVEL=DEBUG` in `.env`

Remember: Athena gets smarter every day! 🧠✨