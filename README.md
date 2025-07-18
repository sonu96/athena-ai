# Athena AI - 24/7 DeFi Agent for Aerodrome

> An autonomous AI agent that monitors, learns, and executes profitable DeFi strategies on Aerodrome (Base blockchain) around the clock.

## 🚀 Overview

Athena is not just another trading bot - she's a learning AI agent with memory, reasoning capabilities, and continuous improvement. Built with cutting-edge AI technologies, she discovers and executes profitable opportunities 24/7 while you sleep.

### Key Features

- **24/7 Autonomous Operation**: Never misses an opportunity
- **Self-Learning**: Improves strategies based on outcomes
- **Memory System**: Remembers patterns, successes, and failures
- **Multi-Strategy Execution**: Swaps, LP, staking, voting, arbitrage
- **Real-time Monitoring**: FastAPI dashboard with live updates
- **Production-Ready**: Deployed on Google Cloud with full observability

## 🏗️ Architecture

```
User Query → LangGraph Agent → Memory (Mem0) → Decision
                    ↓
            CDP Toolkit → Aerodrome → Profit
                    ↓
              Learning Loop
```

### Tech Stack

- **AI/ML**: LangGraph, LangChain, Google Gemini
- **Memory**: Mem0 + Google Firestore
- **Blockchain**: CDP SDK for Base chain
- **API**: FastAPI + WebSockets
- **Observability**: LangSmith
- **Infrastructure**: Google Cloud (Run, Firestore, Pub/Sub, Secret Manager)

## 🚦 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud account with Vertex AI enabled
- CDP API key from Coinbase
- Base chain RPC access (default provided)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/athena-ai.git
cd athena-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your GCP project ID

# Set up secrets (interactive)
python scripts/setup_secrets.py

# Run locally
python run.py
```

### Configuration

```bash
# Required environment variables
GCP_PROJECT_ID=your-project-id

# Secrets (stored in Google Secret Manager)
- cdp-api-key
- cdp-api-secret
- langsmith-api-key (optional)
- mem0-api-key (optional)
```

## 🧠 How Athena Thinks

### Agent States

1. **OBSERVE**: Monitors market conditions, gas prices, pool activity
2. **ANALYZE**: Processes data with historical context from memory
3. **DECIDE**: Evaluates opportunities and selects best strategy
4. **EXECUTE**: Performs on-chain transactions via CDP
5. **LEARN**: Updates memory with outcomes and discovered patterns

### Memory Categories

- **Market Intelligence**: Pool performance, gas patterns, volume trends
- **Strategy Performance**: Success rates, ROI by strategy type
- **Position Tracking**: Current holdings, historical P&L
- **Learned Patterns**: Discovered opportunities and market behaviors

## 📊 Monitoring

### API Endpoints

```bash
# Check health
GET http://localhost:8000/health

# View 24h performance
GET http://localhost:8000/performance/24h

# Current positions
GET http://localhost:8000/positions

# Active strategies
GET http://localhost:8000/strategies/active

# Real-time updates
WS ws://localhost:8000/live
```

### Dashboard

Access the monitoring dashboard via API:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket Live Updates**: ws://localhost:8000/live

## 🚀 Deployment

### Google Cloud Deployment

```bash
# Build and deploy
gcloud builds submit --config deployment/cloudbuild.yaml

# Check deployment
gcloud run services describe athena-agent

# View logs
gcloud logging read "resource.type=cloud_run_revision"
```

### Infrastructure

- **Cloud Run**: Always-on container for agent
- **Firestore**: Persistent memory storage
- **Pub/Sub**: Event-driven architecture
- **Secret Manager**: Secure configuration
- **Cloud Scheduler**: Periodic maintenance tasks

## 🔍 Strategies

### Current Strategies

1. **Liquidity Provision**: Identifies high-yield pools and manages LP positions
2. **Arbitrage**: Detects and executes price discrepancies
3. **Yield Farming**: Optimizes reward claiming and compounding
4. **Vote Optimization**: Maximizes bribes through strategic voting
5. **Gas Optimization**: Executes during low-gas periods

### Learning Examples

```python
# Patterns Athena has discovered:
- "Gas prices drop 40% between 2-5 AM UTC"
- "New pools have 10x higher fees in first hour"
- "Bribes increase 60% before epoch changes"
- "Volume spikes precede major price movements"
```

## 🛡️ Security

- No private keys stored (CDP handles custody)
- All secrets in Google Secret Manager
- Read-only dashboard access
- Transaction limits and risk controls
- Automated security scanning

## 📈 Performance

### Metrics Tracked

- Total profit (24h, 7d, 30d)
- Win rate by strategy
- Gas efficiency
- Learning rate (new patterns discovered)
- Uptime and reliability

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Aerodrome Finance for the amazing protocol
- CDP team for the excellent SDK
- LangChain community for AI tools
- Base chain for low-cost transactions

---

**Remember**: Athena gets smarter every day. The longer she runs, the more profitable she becomes. 🚀