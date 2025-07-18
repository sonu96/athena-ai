# 🔐 Add Your Secrets to Google Secret Manager

Run these commands in your terminal to add your CDP API credentials:

## Step 1: Add CDP API Key (REQUIRED)

```bash
# Replace YOUR_CDP_API_KEY with your actual key
echo -n "YOUR_CDP_API_KEY" | gcloud secrets create cdp-api-key --data-file=- --project=athena-defi-agent-1752635199
```

## Step 2: Add CDP API Secret (REQUIRED)

```bash
# Replace YOUR_CDP_API_SECRET with your actual secret
echo -n "YOUR_CDP_API_SECRET" | gcloud secrets create cdp-api-secret --data-file=- --project=athena-defi-agent-1752635199
```

## Step 3: Add Optional Secrets

### LangSmith (for monitoring - recommended)
```bash
echo -n "YOUR_LANGSMITH_API_KEY" | gcloud secrets create langsmith-api-key --data-file=- --project=athena-defi-agent-1752635199
```

### Mem0 (for cloud memory - optional)
```bash
echo -n "YOUR_MEM0_API_KEY" | gcloud secrets create mem0-api-key --data-file=- --project=athena-defi-agent-1752635199
```

## Step 4: Verify Secrets

```bash
# List all secrets
gcloud secrets list --project=athena-defi-agent-1752635199

# Test reading a secret (be careful not to expose it)
gcloud secrets versions access latest --secret=cdp-api-key --project=athena-defi-agent-1752635199
```

## Where to Get Your Keys

1. **CDP API Keys**: 
   - Go to https://portal.cdp.coinbase.com/
   - Create a new project or use existing
   - Navigate to API Keys section
   - Create new API key with wallet permissions

2. **LangSmith** (Optional but recommended):
   - Go to https://smith.langchain.com/
   - Sign up/Login
   - Go to Settings → API Keys
   - Create new API key

3. **Mem0** (Optional):
   - Go to https://mem0.ai/
   - Sign up for an account
   - Get API key from dashboard

## Example Commands with Placeholders

```bash
# Example with actual format (DO NOT COPY THESE KEYS)
echo -n "sk_live_abcd1234efgh5678ijkl9012mnop3456" | gcloud secrets create cdp-api-key --data-file=- --project=athena-defi-agent-1752635199

echo -n "secret_1234567890abcdefghijklmnopqrs" | gcloud secrets create cdp-api-secret --data-file=- --project=athena-defi-agent-1752635199
```

## After Adding Secrets

Once you've added your secrets, you can run Athena:

```bash
# Install dependencies first
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Athena
python run.py
```

## Troubleshooting

If you get permission errors:
```bash
# Make sure you're authenticated
gcloud auth login

# Set the project
gcloud config set project athena-defi-agent-1752635199
```

If you need to update a secret:
```bash
echo -n "NEW_VALUE" | gcloud secrets versions add SECRET_NAME --data-file=- --project=athena-defi-agent-1752635199
```