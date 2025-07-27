"""
Aerodrome Platform Integration

Provides deep integration with Aerodrome Finance on Base, including:
- Platform mechanics and tokenomics understanding
- Gauge and voting system integration
- Liquidity provision strategies
- Reward optimization
"""

from .platform import AerodromePlatform
from .knowledge_base import AerodromeKnowledgeBase
from .tokenomics import AerodromeTokenomics

__all__ = [
    "AerodromePlatform",
    "AerodromeKnowledgeBase", 
    "AerodromeTokenomics"
]