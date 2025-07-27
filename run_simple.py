"""
Simple run script for Cloud Run deployment
"""
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple FastAPI app for health checks
from fastapi import FastAPI

app = FastAPI(title="Athena AI", description="24/7 DeFi Agent")

@app.get("/")
async def root():
    return {"message": "Athena AI is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "athena-ai"}

@app.get("/status")
async def status():
    return {
        "status": "operational",
        "version": "2.0",
        "features": [
            "QuickNode Aerodrome API integration",
            "Coinbase AgentKit transactions",
            "Platform knowledge system",
            "Risk management"
        ]
    }

if __name__ == "__main__":
    # Use PORT environment variable from Cloud Run, fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting Athena AI on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )