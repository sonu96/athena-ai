# Aerodrome Finance - Whitepaper Summary

## Executive Summary

Aerodrome Finance is a next-generation AMM designed to serve as Base's central liquidity hub, combining the battle-tested ve(3,3) mechanics with a self-optimizing flywheel that aligns protocol emissions with fees generated.

## Core Innovation: ve(3,3) Model

### Vote-Escrow (ve) Mechanics
- **AERO → veAERO**: Lock AERO tokens for 1 week to 4 years
- **Linear Decay**: Voting power decreases linearly until unlock
- **Non-transferable**: veAERO cannot be sold or transferred
- **Maximum Efficiency**: 1 AERO locked for 4 years = 1 veAERO

### (3,3) Game Theory
- **Mutual Benefit**: All participants benefit from cooperation
- **Aligned Incentives**: Voters direct emissions to productive pools
- **Value Accrual**: 100% of fees to participants (0% to protocol)
- **Positive Sum**: Growing TVL benefits all stakeholders

## Token Distribution

### Initial Supply: 500M AERO
- **No pre-mine or insider allocation**
- **Fair launch with initial airdrop**
- **Team tokens vested over 4 years**
- **Community-first distribution**

### Emission Schedule
- **Week 1**: 10M AERO
- **Decay Rate**: 2% per week (exponential decay)
- **Distribution Split**:
  - 67% to Liquidity Providers
  - 33% to veAERO holders (rebase)
- **Terminal State**: Emissions approach zero asymptotically

## Protocol Mechanics

### Liquidity Pools
1. **Stable Pools** (sAMM)
   - For correlated assets (stablecoins)
   - 0.01% trading fee
   - x³y + y³x = k formula
   - Optimized for minimal slippage

2. **Volatile Pools** (vAMM)
   - For uncorrelated assets
   - 0.3% trading fee
   - x × y = k formula
   - Standard constant product

3. **Concentrated Liquidity** (Coming in V3)
   - Capital efficiency improvements
   - Custom fee tiers
   - Active liquidity management

### Voting System
- **Epoch Length**: 1 week (Thursday 00:00 UTC)
- **Vote Allocation**: Distribute 100% of voting power
- **Vote Direction**: Direct emissions to chosen pools
- **Vote Rewards**: Receive fees + bribes from voted pools

### Boost Mechanism
```
Boost = min(1 + 1.5 × (veAERO_share / LP_share), 2.5)
```
- **Minimum Boost**: 1.0x (no veAERO)
- **Maximum Boost**: 2.5x (optimal veAERO/LP ratio)
- **Incentive**: Encourages long-term locking

## Economic Flywheel

1. **Voters** direct emissions to productive pools
2. **Liquidity** flows to high-emission pools
3. **Traders** benefit from deep liquidity
4. **Fees** generated reward voters
5. **Bribes** attract votes to specific pools
6. **Value** accrues to veAERO holders
7. **Cycle** repeats with aligned incentives

## Key Differentiators

### vs. Uniswap V2
- Built-in gauge system
- Native token incentives
- Vote-directed emissions
- Protocol-owned liquidity

### vs. Curve
- Permissionless pool creation
- No admin controls
- Simplified UI/UX
- Optimized for Base

### vs. Velodrome
- Improved contract architecture
- Enhanced capital efficiency
- Better bribe integration
- Refined tokenomics

## Risk Factors

### Smart Contract Risk
- Audited by Zellic
- Battle-tested codebase
- 48-hour timelock
- Bug bounty program

### Economic Risks
- Token inflation from emissions
- Potential vote manipulation
- Liquidity fragmentation
- Competitive landscape

### Mitigation Strategies
- Decaying emissions model
- Sybil-resistant voting
- Incentive alignment
- Continuous innovation

## Future Roadmap

### Phase 1: Foundation (Completed)
- Core AMM deployment
- veAERO implementation
- Gauge system live
- Initial liquidity

### Phase 2: Optimization (Current)
- Concentrated liquidity
- Cross-chain expansion
- Advanced routing
- SDK development

### Phase 3: Ecosystem (Future)
- Lending integration
- Options protocols
- Structured products
- DAO governance

## Conclusion

Aerodrome represents the evolution of DeFi liquidity protocols, combining proven mechanics with innovative tokenomics to create a sustainable, efficient, and user-aligned automated market maker. The ve(3,3) model ensures long-term value accrual while maintaining competitive yields for liquidity providers.

### Key Takeaways
- **Sustainable**: Decaying emissions prevent hyperinflation
- **Efficient**: Vote-directed liquidity goes where needed
- **Aligned**: All participants benefit from growth
- **Decentralized**: No admin keys or central control
- **Innovative**: Continuous improvements and features