# Athena AI API Documentation

## Overview

The Athena AI API provides real-time access to the agent's state, performance metrics, and control capabilities. The API runs alongside the main agent and provides both REST endpoints and WebSocket connections for live updates.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local development. Production deployments should implement proper authentication.

## Endpoints

### Health & Status

#### GET /health
Check if the agent is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "agent_active": true,
  "wallet_address": "0x1234...5678"
}
```

### Performance

#### GET /performance/{period}
Get performance metrics for a specific time period.

**Parameters:**
- `period`: Time period (24h, 7d, 30d)

**Response:**
```json
{
  "period": "24h",
  "total_profit": 847.50,
  "win_rate": 0.85,
  "total_trades": 47,
  "winning_trades": 40,
  "losing_trades": 7,
  "current_positions": [
    {
      "pool": "WETH/USDC",
      "value": 5000.00,
      "apr": 45.2
    }
  ]
}
```

### Positions

#### GET /positions
Get current token balances and positions.

**Response:**
```json
{
  "positions": [
    {
      "token": "WETH",
      "balance": 2.5,
      "usd_value": 5625.00
    },
    {
      "token": "USDC",
      "balance": 10000.0,
      "usd_value": 10000.00
    }
  ],
  "total_value": 15625.00,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Strategies

#### GET /strategies/active
Get currently active strategies and their configuration.

**Response:**
```json
{
  "active_strategies": ["liquidity_provision", "arbitrage", "yield_farming"],
  "strategy_details": {
    "liquidity_provision": {
      "enabled": true,
      "min_apr": 20.0,
      "max_il_tolerance": 0.05
    },
    "arbitrage": {
      "enabled": true,
      "min_profit": 10.0,
      "max_gas_percent": 0.3
    }
  },
  "last_execution": "2024-01-15T10:25:00Z"
}
```

#### POST /strategies/override
Manually override a strategy (emergency control).

**Request Body:**
```json
{
  "strategy": "arbitrage",
  "action": "disable"
}
```

**Response:**
```json
{
  "message": "Strategy arbitrage override: disable",
  "success": true
}
```

### Memory & Learning

#### GET /memories/recent
Get recent memories and patterns discovered by the agent.

**Response:**
```json
{
  "recent_memories": [
    {
      "content": "Gas price: 0.001 gwei at 3:00 UTC",
      "type": "observation",
      "category": "gas_optimization",
      "confidence": 0.9,
      "timestamp": "2024-01-15T03:00:00Z"
    },
    {
      "content": "High APR pool found: AERO/USDC at 89.5% APR",
      "type": "pattern",
      "category": "pool_behavior",
      "confidence": 0.95,
      "timestamp": "2024-01-15T10:00:00Z"
    }
  ],
  "total_memories": 1523,
  "patterns_discovered": 47
}
```

### Market Data

#### GET /gas/recommendation
Get current gas price analysis and recommendations.

**Response:**
```json
{
  "current_price": 0.001,
  "24h_average": 0.0015,
  "optimal_hours": [2, 3, 4, 14, 15],
  "recommendation": "🟢 Excellent time to execute - gas is 20% below average",
  "confidence": 0.87
}
```

#### GET /pools/opportunities
Get current pool opportunities identified by the scanner.

**Query Parameters:**
- `category` (optional): Filter by category (high_apr, high_volume, new_pools, imbalanced)

**Response:**
```json
{
  "opportunities": [
    {
      "pair": "AERO/USDC",
      "apr": 89.5,
      "tvl": 500000,
      "reason": "High APR: 89.5%",
      "score": 0.895
    },
    {
      "pair": "WETH/USDC",
      "volume_24h": 1500000,
      "reason": "High volume: $1,500,000",
      "score": 0.75
    }
  ],
  "summary": {
    "last_scan": "2024-01-15T10:25:00Z",
    "pools_tracked": 15,
    "opportunities": {
      "high_apr": 3,
      "high_volume": 2,
      "new_pools": 0,
      "imbalanced": 1
    },
    "top_apr": 89.5,
    "total_tvl": 25000000
  }
}
```

### WebSocket

#### WS /live
Connect to receive real-time updates about the agent's state.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent update:', data);
};
```

**Message Format:**
```json
{
  "type": "status",
  "timestamp": "2024-01-15T10:30:00Z",
  "emotions": {
    "confidence": 0.8,
    "curiosity": 0.6,
    "caution": 0.3,
    "satisfaction": 0.7
  },
  "performance": {
    "total_profit": 847.50
  },
  "gas": 0.001
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

Common status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Rate Limiting

Currently no rate limiting is implemented. Production deployments should add appropriate rate limits.

## Examples

### Python
```python
import requests

# Get current performance
response = requests.get('http://localhost:8000/performance/24h')
data = response.json()
print(f"24h Profit: ${data['total_profit']}")

# Get gas recommendation
response = requests.get('http://localhost:8000/gas/recommendation')
gas_data = response.json()
print(f"Gas recommendation: {gas_data['recommendation']}")
```

### JavaScript
```javascript
// Fetch current positions
fetch('http://localhost:8000/positions')
  .then(res => res.json())
  .then(data => {
    console.log(`Total portfolio value: $${data.total_value}`);
    data.positions.forEach(pos => {
      console.log(`${pos.token}: ${pos.balance} ($${pos.usd_value})`);
    });
  });

// Connect to WebSocket for live updates
const ws = new WebSocket('ws://localhost:8000/live');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  document.getElementById('profit').textContent = `$${update.performance.total_profit}`;
  document.getElementById('confidence').textContent = `${(update.emotions.confidence * 100).toFixed(0)}%`;
};
```

## Risk Management Endpoints

### GET /risk/status
Get current risk metrics and circuit breaker status.

**Response:**
```json
{
  "portfolio_risk_score": 35,
  "risk_level": "MEDIUM",
  "circuit_breakers": {
    "rapid_loss": {
      "status": "active",
      "triggered_at": null,
      "cooldown_until": null
    },
    "portfolio_drawdown": {
      "status": "active",
      "triggered_at": null
    },
    "gas_manipulation": {
      "status": "tripped",
      "triggered_at": "2024-01-15T10:30:00Z",
      "cooldown_until": "2024-01-15T11:00:00Z"
    }
  },
  "exposure_summary": {
    "max_position_concentration": 0.18,
    "token_exposures": {
      "WETH": 0.35,
      "USDC": 0.40,
      "AERO": 0.25
    }
  }
}
```

### POST /risk/override
Override risk limits (requires admin authentication).

**Request Body:**
```json
{
  "parameter": "max_position_size",
  "value": 2000,
  "duration_minutes": 60,
  "reason": "High confidence opportunity"
}
```

### GET /risk/exposure
Get detailed portfolio exposure analysis.

## Performance Endpoints

### GET /performance/cache-stats
Get cache hit rates and performance metrics.

**Response:**
```json
{
  "caches": {
    "pattern_cache": {
      "size": 847,
      "max_size": 1000,
      "hit_rate": 0.73,
      "ttl": 3600
    },
    "query_cache": {
      "size": 423,
      "max_size": 500,
      "hit_rate": 0.65,
      "ttl": 300
    },
    "llm_cache": {
      "size": 512,
      "max_size": 1000,
      "hit_rate": 0.81,
      "ttl": 86400
    }
  },
  "total_hit_rate": 0.73
}
```

### GET /performance/query-stats
Get query performance statistics.

**Response:**
```json
{
  "memory_queries": {
    "count_24h": 15420,
    "avg_latency_ms": 85,
    "p95_latency_ms": 142,
    "p99_latency_ms": 203
  },
  "firestore_operations": {
    "reads_per_minute": 45,
    "writes_per_minute": 12,
    "batch_efficiency": 0.82
  }
}
```

## Recovery Endpoints

### POST /recovery/checkpoint
Manually trigger state checkpoint.

**Response:**
```json
{
  "checkpoint_id": "ckpt_20240115_103500",
  "timestamp": "2024-01-15T10:35:00Z",
  "storage_locations": ["firestore", "gcs", "local"],
  "size_mb": 2.4,
  "checksum": "sha256:abcd1234..."
}
```

### POST /recovery/restore
Restore from specific checkpoint.

**Request Body:**
```json
{
  "checkpoint_id": "ckpt_20240115_093000",
  "verify": true
}
```

### GET /recovery/status
Get recovery system status.

**Response:**
```json
{
  "recovery_mode": false,
  "last_checkpoint": {
    "id": "ckpt_20240115_103000",
    "age_minutes": 5,
    "verified": true
  },
  "available_checkpoints": 168,
  "backup_health": {
    "firestore": "healthy",
    "gcs": "healthy",
    "local": "healthy"
  }
}
```

## Monitoring Endpoints

### GET /metrics
Prometheus-compatible metrics endpoint.

**Response (Prometheus format):**
```
# HELP memory_growth_rate Number of new memories per hour
# TYPE memory_growth_rate gauge
memory_growth_rate 85.0

# HELP cache_hit_rate Cache hit rate ratio
# TYPE cache_hit_rate gauge
cache_hit_rate{cache="pattern"} 0.73
cache_hit_rate{cache="query"} 0.65

# HELP circuit_breaker_triggers Total circuit breaker activations
# TYPE circuit_breaker_triggers counter
circuit_breaker_triggers{type="rapid_loss"} 2
circuit_breaker_triggers{type="gas_manipulation"} 5
```

### GET /health/detailed
Comprehensive health check of all subsystems.

**Response:**
```json
{
  "status": "healthy",
  "subsystems": {
    "memory": {
      "status": "healthy",
      "memory_count": 4523,
      "query_latency_ms": 82
    },
    "firestore": {
      "status": "healthy",
      "write_queue_depth": 3,
      "last_error": null
    },
    "mcp": {
      "status": "healthy",
      "last_query_ms": 145
    },
    "agentkit": {
      "status": "healthy",
      "wallet_accessible": true
    }
  },
  "uptime_seconds": 432000,
  "last_cycle": "2024-01-15T10:35:00Z"
}
```

## Running the API

The API can be run standalone or alongside the main agent:

```bash
# Standalone
python -m src.api.main

# Or with custom settings
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

When running with the main agent, the API is automatically started and connected to the agent components.