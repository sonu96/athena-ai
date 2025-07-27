# Disaster Recovery Procedures

## Overview

This document outlines comprehensive disaster recovery procedures for Athena AI, covering system failures, data corruption, and emergency situations. All procedures are designed to minimize downtime and data loss while ensuring safe recovery.

## Recovery Objectives

- **Recovery Time Objective (RTO)**: < 10 minutes for critical failures
- **Recovery Point Objective (RPO)**: < 1 hour maximum data loss
- **Uptime Target**: 99.9% (8.76 hours downtime/year)

## State Checkpoint System

### Automatic Checkpoints

The system creates hourly snapshots of critical state:

```python
# Checkpoint Configuration
CHECKPOINT_CONFIG = {
    "interval": timedelta(hours=1),
    "retention_days": 7,
    "storage_locations": [
        "firestore",      # Primary
        "gcs",           # Backup
        "local_disk"     # Emergency
    ],
    "compression": "gzip"
}
```

### Checkpoint Contents

```json
{
    "timestamp": "2024-01-15T10:00:00Z",
    "version": "2.0",
    "checksum": "sha256:abcd1234...",
    "state": {
        "positions": {
            "pool_addresses": ["0x..."],
            "lp_tokens": {...},
            "values_usd": {...}
        },
        "critical_memories": {
            "patterns": [...],
            "high_confidence_pools": [...],
            "recent_decisions": [...]
        },
        "performance_metrics": {
            "total_value": 10000.50,
            "profit_24h": 125.30,
            "gas_spent": 45.20
        },
        "risk_state": {
            "portfolio_risk_score": 35,
            "active_circuit_breakers": [],
            "exposure_summary": {...}
        },
        "configuration": {
            "risk_limits": {...},
            "strategy_params": {...}
        }
    }
}
```

### Creating Manual Checkpoint

```bash
# Force checkpoint creation
curl -X POST localhost:8000/recovery/checkpoint \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Verify checkpoint
curl localhost:8000/recovery/checkpoints/latest
```

## Recovery Procedures

### 1. Memory Corruption Recovery

**Symptoms:**
- Inconsistent memory recalls
- Vector search errors
- Checksum validation failures

**Recovery Steps:**

```bash
# 1. Stop the agent immediately
systemctl stop athena-agent

# 2. Run corruption detection
python scripts/detect_memory_corruption.py

# Output example:
# Corrupted memories found: 47
# Categories affected: ['pattern', 'pool_behavior']
# Corruption type: metadata_mismatch

# 3. Backup current state (even if corrupted)
python scripts/backup_corrupted_state.py --output=/backup/corrupted_$(date +%Y%m%d_%H%M%S)

# 4. Run recovery script
python scripts/recover_memory.py --source=firestore --validate

# 5. Rebuild vector indices
python scripts/rebuild_mem0_indices.py

# 6. Validate recovery
python scripts/validate_memory_integrity.py

# 7. Resume with reduced confidence
python run.py --recovery-mode --initial-confidence=0.5
```

**Recovery Script Details:**

```python
# scripts/recover_memory.py
async def recover_memory_system():
    """Full memory recovery procedure"""
    
    # 1. Identify corruption extent
    corrupted = await detect_corrupted_memories()
    logger.info(f"Found {len(corrupted)} corrupted memories")
    
    # 2. Quarantine corrupted data
    await quarantine_memories(corrupted)
    
    # 3. Restore from Firestore backup
    clean_memories = await restore_from_firestore()
    
    # 4. Rebuild Mem0 database
    await mem0.clear()
    for memory in clean_memories:
        await mem0.add(memory)
    
    # 5. Reindex for vector search
    await mem0.reindex()
    
    # 6. Verify integrity
    if not await verify_integrity():
        raise RecoveryError("Integrity check failed")
    
    logger.info("Memory recovery completed successfully")
```

### 2. Position Mismatch Recovery

**Symptoms:**
- Blockchain state differs from local records
- Missing or extra positions
- Value discrepancies > $100

**Recovery Steps:**

```bash
# 1. Pause trading
curl -X POST localhost:8000/pause

# 2. Run position reconciliation
python scripts/reconcile_positions.py

# Output example:
# Blockchain positions: 5
# Firestore positions: 4
# Discrepancies found:
#   - Missing: WETH/USDC (0x...)
#   - Value mismatch: AERO/USDC ($1250 vs $1180)

# 3. Review discrepancies
python scripts/reconcile_positions.py --detailed > position_report.txt

# 4. For discrepancies > $1000, manual review required
# Check position_report.txt and verify on Etherscan

# 5. Update state to match blockchain (source of truth)
python scripts/update_positions.py --source=blockchain --confirm

# 6. Verify wallet balance
python scripts/verify_wallet.py

# 7. Resume trading
curl -X POST localhost:8000/resume
```

**Reconciliation Logic:**

```python
# scripts/reconcile_positions.py
async def reconcile_positions():
    """Compare and reconcile position state"""
    
    # 1. Get blockchain positions (source of truth)
    blockchain_positions = await agentkit.get_all_positions()
    
    # 2. Get database positions
    db_positions = await firestore.get_positions()
    
    # 3. Compare and identify discrepancies
    discrepancies = []
    
    # Check for missing positions
    for bp in blockchain_positions:
        if bp.address not in [p.address for p in db_positions]:
            discrepancies.append({
                "type": "missing_in_db",
                "position": bp
            })
    
    # Check for extra positions
    for dp in db_positions:
        if dp.address not in [p.address for p in blockchain_positions]:
            discrepancies.append({
                "type": "extra_in_db",
                "position": dp
            })
    
    # Check for value mismatches
    for bp in blockchain_positions:
        dp = next((p for p in db_positions if p.address == bp.address), None)
        if dp and abs(bp.value_usd - dp.value_usd) > 100:
            discrepancies.append({
                "type": "value_mismatch",
                "blockchain": bp,
                "database": dp
            })
    
    return discrepancies
```

### 3. Total System Failure Recovery

**Symptoms:**
- Agent completely unresponsive
- Multiple subsystem failures
- Corruption across multiple components

**Recovery Steps:**

```bash
# 1. Verify wallet integrity first (critical!)
python scripts/verify_wallet_integrity.py

# Output:
# Wallet address: 0x...
# Can sign: ✓
# Balance check: ✓

# 2. Find latest valid checkpoint
python scripts/find_latest_checkpoint.py

# Output:
# Latest checkpoint: 2024-01-15T09:00:00Z
# Location: gs://athena-backups/checkpoints/...
# Integrity: VALID

# 3. Restore from checkpoint
python scripts/restore_checkpoint.py \
  --checkpoint=2024-01-15T09:00:00Z \
  --verify

# 4. Replay recent events
python scripts/replay_events.py \
  --from=2024-01-15T09:00:00Z \
  --source=blockchain

# 5. Start in observation mode
export OBSERVATION_MODE=true
export MIN_PATTERN_CONFIDENCE=0.8
python run.py

# 6. Monitor for 1 hour
tail -f logs/athena.log | grep -E "ERROR|WARNING|CRITICAL"

# 7. Gradually increase confidence
# After 1 hour:
export MIN_PATTERN_CONFIDENCE=0.7
# After 3 hours:
export MIN_PATTERN_CONFIDENCE=0.6
# After 6 hours:
export OBSERVATION_MODE=false
```

### 4. API Service Failures

**Symptoms:**
- QuickNode MCP timeout
- AgentKit connection errors
- Firestore unavailable

**Recovery Steps:**

```bash
# 1. Check service status
python scripts/check_services.py

# 2. Activate fallback mode
export FALLBACK_MODE=true
python run.py

# Fallback behaviors:
# - QuickNode MCP → Cached data + basic RPC
# - AgentKit → Queue transactions
# - Firestore → Local SQLite cache

# 3. Monitor service recovery
python scripts/monitor_service_recovery.py

# 4. Resume normal mode when services recover
export FALLBACK_MODE=false
systemctl restart athena-agent
```

## Emergency Response Procedures

### Circuit Breaker Activation

When a circuit breaker trips:

```python
# Automatic response flow
async def handle_circuit_breaker(breaker_type, context):
    """Automated circuit breaker response"""
    
    if breaker_type == "RAPID_LOSS":
        # 1. Log the event
        logger.critical(f"Circuit breaker: {breaker_type}")
        
        # 2. Pause operations
        await trading_engine.pause()
        
        # 3. Assess positions
        risky_positions = await assess_risk()
        
        # 4. Emergency exit if needed
        if context.severity == "CRITICAL":
            await emergency_exit_positions(risky_positions)
        
        # 5. Wait for cooldown
        await asyncio.sleep(breaker.cooldown)
        
        # 6. Resume cautiously
        await trading_engine.resume(confidence=0.5)
```

### Manual Emergency Controls

```bash
# Emergency stop - halts everything immediately
curl -X POST localhost:8000/emergency/stop \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Force exit all positions
curl -X POST localhost:8000/emergency/exit-all \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"confirm": true}'

# Override circuit breaker
curl -X POST localhost:8000/emergency/override \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "breaker": "RAPID_LOSS",
    "action": "reset",
    "reason": "Market conditions normalized"
  }'
```

## Monitoring & Alerts

### Health Check Endpoints

```bash
# Basic health
curl localhost:8000/health

# Detailed system status
curl localhost:8000/health/detailed

# Recovery system status
curl localhost:8000/recovery/status
```

### Alert Configuration

```yaml
# alerts.yaml
alerts:
  - name: MemoryCorruption
    condition: memory_checksum_fail
    channels: ["pagerduty", "slack"]
    severity: critical
    
  - name: PositionMismatch
    condition: position_discrepancy > 1000
    channels: ["slack", "email"]
    severity: high
    
  - name: CheckpointFailed
    condition: checkpoint_creation_error
    channels: ["slack"]
    severity: medium
    
  - name: RecoveryInProgress
    condition: recovery_mode_active
    channels: ["slack"]
    severity: info
```

## Testing Recovery Procedures

### Monthly Disaster Recovery Drill

```bash
# 1. Create test scenario
python scripts/dr_drill.py --scenario=memory_corruption

# 2. Execute recovery
./recovery_procedures/memory_corruption.sh

# 3. Validate results
python scripts/validate_dr_drill.py

# 4. Document results
echo "DR Drill $(date): " >> dr_drill_log.txt
```

### Backup Verification

```bash
# Daily backup verification
0 6 * * * /opt/athena/scripts/verify_backups.py

# Weekly checkpoint restoration test
0 3 * * 0 /opt/athena/scripts/test_checkpoint_restore.py
```

## Recovery Automation Scripts

### Key Scripts

1. **detect_memory_corruption.py** - Identifies corrupted memories
2. **recover_memory.py** - Restores memory from backup
3. **reconcile_positions.py** - Compares blockchain vs database
4. **restore_checkpoint.py** - Restores from checkpoint
5. **verify_wallet_integrity.py** - Ensures wallet is accessible
6. **emergency_exit.py** - Closes all positions immediately

### Script Templates

```python
# Template for recovery scripts
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def main():
    """Recovery procedure template"""
    try:
        # 1. Pre-recovery checks
        if not await pre_recovery_checks():
            logger.error("Pre-recovery checks failed")
            return False
            
        # 2. Backup current state
        backup_path = await backup_current_state()
        logger.info(f"State backed up to: {backup_path}")
        
        # 3. Execute recovery
        recovery_result = await execute_recovery()
        
        # 4. Validate recovery
        if await validate_recovery():
            logger.info("Recovery completed successfully")
            return True
        else:
            logger.error("Recovery validation failed")
            await rollback_recovery(backup_path)
            return False
            
    except Exception as e:
        logger.critical(f"Recovery failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())
```

## Contact Information

### Emergency Contacts

- **Primary On-Call**: [Phone/Slack]
- **Secondary On-Call**: [Phone/Slack]
- **Engineering Lead**: [Email/Phone]
- **DevOps Team**: [Slack Channel]

### Escalation Path

1. On-call engineer (0-15 minutes)
2. Engineering lead (15-30 minutes)
3. CTO (30+ minutes)

## Post-Recovery Procedures

### Recovery Report

After any recovery event:

1. Document the incident
2. Analyze root cause
3. Update procedures if needed
4. Share learnings with team

### Performance Validation

```python
# Verify system performance post-recovery
async def validate_performance():
    checks = {
        "memory_query_latency": lambda: test_memory_performance(),
        "trading_execution": lambda: test_trading_capability(),
        "risk_calculations": lambda: test_risk_system(),
        "api_responsiveness": lambda: test_api_endpoints()
    }
    
    results = {}
    for check, func in checks.items():
        results[check] = await func()
        
    return all(results.values())
```

## Appendix: Common Issues

### Issue: Mem0 Index Corruption
**Solution**: Delete indices and rebuild
```bash
rm -rf /var/lib/mem0/indices/*
python scripts/rebuild_mem0_indices.py
```

### Issue: Firestore Quota Exceeded
**Solution**: Activate read cache and reduce query rate
```bash
export FIRESTORE_CACHE_MODE=aggressive
export FIRESTORE_QUERY_LIMIT=100
```

### Issue: Wallet Access Lost
**Solution**: Restore from encrypted backup
```bash
python scripts/restore_wallet.py --source=secret-manager
```