#!/bin/bash

# Script to add secrets to Google Secret Manager

echo "🔐 Adding Secrets to Google Secret Manager"
echo "=========================================="

PROJECT_ID="athena-defi-agent-1752635199"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to create or update a secret
add_secret() {
    local secret_name=$1
    local secret_value=$2
    
    # Check if secret exists
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &>/dev/null; then
        echo -e "${YELLOW}Secret '$secret_name' already exists. Updating...${NC}"
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=- --project="$PROJECT_ID"
        echo -e "${GREEN}✓ Updated secret: $secret_name${NC}"
    else
        echo -e "Creating new secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets create "$secret_name" --data-file=- --project="$PROJECT_ID"
        echo -e "${GREEN}✓ Created secret: $secret_name${NC}"
    fi
}

# CDP API Credentials
echo -e "\n${YELLOW}CDP API Credentials${NC}"
echo "Get your API keys from: https://portal.cdp.coinbase.com/"
echo

read -p "Enter your CDP API Key: " CDP_KEY
read -s -p "Enter your CDP API Secret: " CDP_SECRET
echo

if [ -n "$CDP_KEY" ] && [ -n "$CDP_SECRET" ]; then
    add_secret "cdp-api-key" "$CDP_KEY"
    add_secret "cdp-api-secret" "$CDP_SECRET"
else
    echo -e "${RED}CDP credentials are required!${NC}"
    exit 1
fi

# Optional: LangSmith API Key
echo -e "\n${YELLOW}LangSmith API Key (Optional)${NC}"
echo "Get your API key from: https://smith.langchain.com/"
read -p "Enter LangSmith API Key (or press Enter to skip): " LANGSMITH_KEY

if [ -n "$LANGSMITH_KEY" ]; then
    add_secret "langsmith-api-key" "$LANGSMITH_KEY"
fi

# Optional: Mem0 API Key
echo -e "\n${YELLOW}Mem0 API Key (Optional)${NC}"
echo "Get your API key from: https://mem0.ai/"
read -p "Enter Mem0 API Key (or press Enter to skip): " MEM0_KEY

if [ -n "$MEM0_KEY" ]; then
    add_secret "mem0-api-key" "$MEM0_KEY"
fi

# List all secrets
echo -e "\n${GREEN}✅ Secrets configured:${NC}"
gcloud secrets list --project="$PROJECT_ID" --format="table(name,created)"

echo -e "\n${GREEN}🎉 Setup complete! You can now run:${NC}"
echo "python3 run.py"