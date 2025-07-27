# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Athena is a 24/7 autonomous DeFi agent for Aerodrome on Base blockchain. It uses LangGraph for AI orchestration, Google Cloud Platform for infrastructure, QuickNode's Aerodrome API for market data, and Coinbase AgentKit for AI-native transaction execution.

## Common Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

# Run with Docker Compose (includes Redis & Qdrant)
docker-compose up
```

### Testing
```bash
# Run tests
pytest

# Run async tests
pytest --asyncio-mode=auto
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type check
mypy .

# Run pre-commit hooks
pre-commit run --all-files
```

### Deployment
```bash
# Deploy to Cloud Run
gcloud run deploy athena-ai --source . --region us-central1

# Deploy via Cloud Build
gcloud builds submit --config deployment/cloudbuild.yaml

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=athena-ai" --limit 50
```

## High-Level Architecture

### Platform Knowledge System (NEW)
Athena now incorporates deep platform understanding:
- **Knowledge Base**: Loads documentation, whitepapers, and strategies
- **Tokenomics Model**: Understands emission schedules and voting dynamics
- **Strategy Validation**: Validates actions against platform rules
- **Opportunity Discovery**: Finds opportunities based on platform mechanics

See `src/platforms/` for implementation and `docs/PLATFORM_KNOWLEDGE.md` for details.

### Core Components

1. **Agent System** (`src/agent/`)
   - `core.py`: LangGraph state machine implementation
   - States: OBSERVE → ANALYZE → DECIDE → EXECUTE → LEARN
   - Uses Google Gemini for reasoning
   - Emotional modeling for decision confidence

2. **Memory Architecture** (`src/agent/memory.py`)
   - Hierarchical: Short-term (Redis), Long-term (Firestore), Semantic (Vector DB)
   - Categories: observations, pool_analysis, gas_patterns, strategies, decisions, emotions
   - Pattern recognition and learning capabilities

3. **Blockchain Integration**
   - **QuickNode Aerodrome API** (`src/collectors/aerodrome_api_scanner.py`): Market data via specialized API endpoints
   - **Coinbase AgentKit** (`src/agentkit/`): AI-native transaction execution
   - **CDP RPC**: Optional authenticated RPC endpoint (configured but primarily for transactions)
   - Secure wallet management using existing CDP credentials

4. **Data Collection Pipeline** (`src/collectors/`)
   - `pool_scanner.py`: Monitors Aerodrome pools
   - `gas_monitor.py`: Tracks gas prices
   - Event-driven via Google Pub/Sub

5. **API Layer** (`src/api/`)
   - FastAPI with WebSocket support
   - Real-time updates and monitoring
   - Endpoints: `/health`, `/performance/24h`, `/positions`, `/gas/recommendation`

### Configuration Management

- **Settings**: `config/settings.py` - Pydantic models with GCP Secret Manager integration
- **Contracts**: `config/contracts.py` - Smart contract addresses
- **Environment**: `.env` file for local development (GCP_PROJECT_ID required)

### Security Architecture

- All secrets stored in Google Secret Manager
- CDP private keys never stored locally
- Transaction simulation before execution
- Risk limits and emergency controls

### Deployment Architecture

- **Cloud Run**: Always-on service with health checks
- **Firestore**: State and memory persistence
- **Pub/Sub**: Event-driven communication
- **Secret Manager**: Secure credential storage

## Key Development Patterns

### Working with the Agent
- Agent decisions flow through the LangGraph state machine
- Each state has specific responsibilities and outputs
- Emotional states influence decision confidence
- Memory retrieval informs analysis

### Adding New Strategies
1. Update `config/settings.py` with strategy parameters
2. Implement strategy logic in agent states
3. Add memory categories if needed
4. Update decision logic in `src/agent/core.py`

### Blockchain Interactions
- Use QuickNode Aerodrome API for pool data queries (APR, TVL, volume)
- Use Coinbase AgentKit for all transactions
- CDP RPC available as authenticated endpoint but not required for data collection
- Note: Gauge reward data collection is not currently implemented

### Blockchain Integration

#### QuickNode Aerodrome API (Data Queries)
```python
from src.collectors.aerodrome_api_scanner import AerodromeAPIScanner

# Initialize API scanner
scanner = AerodromeAPIScanner(memory)

# API queries for pool data
pools = await scanner.scan_pools()  # Get pools via QuickNode's Aerodrome API
pool_data = await scanner.get_pool_details(pool_address)  # Specific pool data
# Note: Gas prices use standard RPC calls, gauge rewards not implemented
```

#### Coinbase AgentKit (Transactions)
```python
from src.agentkit.agent_client import AthenaAgentKit

# Initialize with existing CDP credentials
agentkit = AthenaAgentKit()
await agentkit.initialize()

# Natural language execution
result = await agentkit.execute_natural_language(
    "Swap 100 USDC for WETH with 0.5% slippage on Aerodrome"
)

# Or use specific methods
tx = await agentkit.swap("USDC", "WETH", Decimal("100"), slippage=0.5)
tx = await agentkit.add_liquidity("USDC", "WETH", amount_a=1000, amount_b=0.5)
```

#### Configuration
- **QuickNode Endpoint**: API endpoint for Aerodrome data (uses addon/1051/v1)
- **CDP API Key/Secret**: Used by AgentKit for wallet operations
- **CDP RPC URL**: Optional authenticated RPC (constructed but not actively used)
- **Wallet Management**: AgentKit handles wallet creation/recovery
- API calls for pool data, AgentKit for transactions

### API Development
- Follow FastAPI patterns
- Use dependency injection (`src/api/dependencies.py`)
- Implement proper error handling
- Add WebSocket support for real-time features

## Important Notes

- The project runs continuously (24/7) in production
- Always test with simulation mode first
- Monitor costs via GCP billing alerts
- Check logs regularly for agent decisions
- AgentKit uses existing CDP credentials - no new registration needed
- QuickNode Aerodrome API provides specialized pool data endpoints
- CDP RPC configured but primarily for transaction execution
- **Important**: Gauge reward data collection is NOT currently implemented

## Recent Updates (July 2025)

### Observation Mode
Athena now starts in observation mode to learn market patterns before trading:

#### Configuration
```bash
# Enable observation mode (default: true)
export OBSERVATION_MODE=true

# Set observation period in days (default: 3)
export OBSERVATION_DAYS=3

# Minimum pattern confidence to act on (default: 0.7)
export MIN_PATTERN_CONFIDENCE=0.7
```

#### How It Works
1. **Pattern Collection**: During observation, the agent:
   - Monitors market conditions 24/7
   - Identifies time-based patterns (hourly, daily)
   - Tracks gas price correlations
   - Discovers pool APR fluctuations
   - Records arbitrage opportunities

2. **Pattern Storage**: Patterns are stored in Firestore with:
   - Pattern type and description
   - Time context (hour, day)
   - Market conditions when discovered
   - Confidence scores that improve over time

3. **Transition to Trading**: After observation period:
   - High-confidence patterns guide initial trades
   - Conservative parameters for pattern-based decisions
   - Continuous learning and pattern refinement

4. **Monitoring Progress**:
   - Check `observation_metrics` collection in Firestore
   - View discovered patterns in `observed_patterns`
   - Track confidence scores in `pattern_confidence`

### Aerodrome V2 Support
- Added fallback to storage slot reading when getReserves() fails
- Updated pool addresses to official Aerodrome V2 pools
- Enhanced decimal handling for different token configurations

### API Keys Configuration
- CDP Client API Key required for authenticated RPC access
- Google AI API key uses Gemini 1.5 Flash model
- Mem0 Pro plan required for graph memories feature

### Enhanced Memory System (v2.0)
The memory system has been significantly enhanced to store comprehensive pool data:

#### Key Improvements
1. **Comprehensive Storage**: Stores 10-15 memories per scan (vs 2 in v1.0)
   - All pools with APR >= 20%
   - All pools with volume >= $100k
   - All significantly imbalanced pools

2. **Pool Profiles**: Individual behavior tracking per pool
   - Historical ranges (APR, TVL, volume)
   - Time patterns (hourly, daily)
   - Predictive capabilities

3. **New Firestore Collections**:
   - `pool_profiles`: Individual pool behaviors
   - `pool_metrics`: Time-series pool data
   - `pattern_correlations`: Cross-pool relationships

#### Configuration
```bash
# Memory system thresholds
export MIN_APR_FOR_MEMORY=20          # Store pools with APR >= 20%
export MIN_VOLUME_FOR_MEMORY=100000   # Store pools with volume >= $100k
export MAX_MEMORIES_PER_CYCLE=50      # Prevent memory overflow
export POOL_PROFILE_UPDATE_INTERVAL=3600  # Update profiles every hour
```

#### Usage
```python
# Pool-specific memory queries
memories = await memory.recall_pool_memories("WETH/USDC", hours=24)

# Get pool predictions
predictions = pool_profiles.predict_opportunities(next_hour)

# Find cross-pool correlations
correlations = await memory.get_cross_pool_correlations()
```

### Real Data Collection (July 2025)
Athena now collects real market data from Aerodrome - NO HARDCODED VALUES:

#### Components
1. **Gauge System Knowledge** (`src/platforms/knowledge/aerodrome/gauge_system.json`)
   - Contains gauge system information as knowledge base data
   - **Note**: Direct gauge reward collection is NOT currently implemented
   - Pool APR data comes from QuickNode Aerodrome API, not gauge contracts
   - Future implementation would need gauge contract integration

2. **Event Monitor** (`src/aerodrome/event_monitor.py`)
   - Tracks Swap events for real volume data
   - Monitors Fee events for fee collection
   - Maintains hourly/daily volume history
   - Returns 0 volume if no events found (no estimates)

3. **Enhanced Pool Scanner** (`src/collectors/pool_scanner.py`)
   - **REMOVED**: All hardcoded APR estimates
   - **REMOVED**: Volume estimation logic
   - Fetches real AERO price from AERO/USDC pool
   - Uses only real gauge emissions (0 if unavailable)
   - Calculates fee APR from actual 24h volume only
   - Stores only verified data in memory

#### Memory Categories (Current Implementation)
- `pool_analysis`: Pool data from QuickNode Aerodrome API
- `volume_tracking`: Volume data from API (not events currently)
- `arbitrage_opportunity`: Detected price imbalances
- `new_pool`: New pool discoveries
- `apr_anomaly`: Unusual APR changes
- `fee_collection`: Fee tracking (planned, not implemented)
- **Note**: `gauge_emissions` category exists but gauge data not actively collected

#### Testing Real Data
```bash
# Test the real data collection pipeline
python scripts/test_real_data.py
```

#### Current Data Sources and Limitations
- **QuickNode Aerodrome API**: Primary source for pool APR, TVL, and volume data
- **Event Monitoring**: Not currently implemented (planned feature)
- **Gauge Data**: Knowledge base only - direct gauge contract reading not implemented
- **AERO Price**: Fetched via API, not direct contract calls
- **Transaction Execution**: Via AgentKit using CDP credentials

## Current Implementation Status (Updated July 2025)

### What's Currently Working
- ✅ QuickNode Aerodrome API integration for pool data
- ✅ Coinbase AgentKit for transaction execution  
- ✅ Platform knowledge system with Aerodrome documentation
- ✅ Memory system with Firestore persistence
- ✅ Observation mode and pattern recognition
- ✅ Risk management and circuit breakers

### What's Planned/In Development
- ⏳ Direct gauge contract reading for emission data
- ⏳ Event monitoring for real-time volume tracking
- ⏳ Direct Web3 integration for custom contract calls
- ⏳ Real-time fee collection monitoring

### Data Flow Architecture
```
QuickNode Aerodrome API → Pool Data (APR, TVL, Volume)
Platform Knowledge Base → Strategy Validation
Coinbase AgentKit → Transaction Execution
Firestore → Memory & State Persistence
```

### Known Issues
- Some Aerodrome V2 pools don't implement standard Uniswap V2 interface
- Gauge emission data relies on API rather than direct contract calls
- Event monitoring not yet implemented for real-time volume tracking
- Mem0 API may show "free plan" errors if API key is not properly configured