"""
Aerodrome Protocol Contract Addresses and ABIs
"""

# Base Mainnet Addresses
CONTRACTS = {
    "router": {
        "address": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
        "abi": [
            {
                "name": "swapExactTokensForTokens",
                "type": "function",
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "routes", "type": "tuple[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "outputs": [{"name": "amounts", "type": "uint256[]"}]
            },
            {
                "name": "addLiquidity",
                "type": "function",
                "inputs": [
                    {"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                    {"name": "stable", "type": "bool"},
                    {"name": "amountADesired", "type": "uint256"},
                    {"name": "amountBDesired", "type": "uint256"},
                    {"name": "amountAMin", "type": "uint256"},
                    {"name": "amountBMin", "type": "uint256"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "outputs": [
                    {"name": "amountA", "type": "uint256"},
                    {"name": "amountB", "type": "uint256"},
                    {"name": "liquidity", "type": "uint256"}
                ]
            },
            {
                "name": "getAmountsOut",
                "type": "function",
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "routes", "type": "tuple[]"}
                ],
                "outputs": [{"name": "amounts", "type": "uint256[]"}]
            }
        ]
    },
    "factory": {
        "address": "0x420DD381b31aEf6683db6B902084cB0FFECe40Da",
        "abi": [
            {
                "name": "getPool",
                "type": "function",
                "inputs": [
                    {"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                    {"name": "stable", "type": "bool"}
                ],
                "outputs": [{"name": "pool", "type": "address"}]
            },
            {
                "name": "allPairsLength",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "uint256"}]
            }
        ]
    },
    "voter": {
        "address": "0x16613524e02ad97eDfeF371bC883F2F5d6C480A5",
        "abi": [
            {
                "name": "vote",
                "type": "function",
                "inputs": [
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "pools", "type": "address[]"},
                    {"name": "weights", "type": "uint256[]"}
                ]
            },
            {
                "name": "claimBribes",
                "type": "function",
                "inputs": [
                    {"name": "bribes", "type": "address[]"},
                    {"name": "tokens", "type": "address[][]"},
                    {"name": "tokenId", "type": "uint256"}
                ]
            },
            {
                "name": "gauges",
                "type": "function",
                "inputs": [
                    {"name": "pool", "type": "address"}
                ],
                "outputs": [
                    {"name": "gauge", "type": "address"}
                ],
                "stateMutability": "view"
            },
            {
                "name": "distribute",
                "type": "function",
                "inputs": [
                    {"name": "gauges", "type": "address[]"}
                ]
            }
        ]
    },
    "gauge": {
        "abi": [
            {
                "name": "deposit",
                "type": "function",
                "inputs": [
                    {"name": "amount", "type": "uint256"},
                    {"name": "tokenId", "type": "uint256"}
                ]
            },
            {
                "name": "withdraw",
                "type": "function",
                "inputs": [
                    {"name": "amount", "type": "uint256"}
                ]
            },
            {
                "name": "getReward",
                "type": "function",
                "inputs": [
                    {"name": "account", "type": "address"},
                    {"name": "tokens", "type": "address[]"}
                ]
            },
            {
                "name": "earned",
                "type": "function",
                "inputs": [
                    {"name": "account", "type": "address"},
                    {"name": "token", "type": "address"}
                ],
                "outputs": [
                    {"name": "amount", "type": "uint256"}
                ],
                "stateMutability": "view"
            },
            {
                "name": "rewardRate",
                "type": "function",
                "inputs": [
                    {"name": "token", "type": "address"}
                ],
                "outputs": [
                    {"name": "rate", "type": "uint256"}
                ],
                "stateMutability": "view"
            },
            {
                "name": "totalSupply",
                "type": "function",
                "outputs": [
                    {"name": "supply", "type": "uint256"}
                ],
                "stateMutability": "view"
            }
        ]
    },
    "pool": {
        "abi": [
            {
                "name": "getReserves",
                "type": "function",
                "inputs": [],
                "outputs": [
                    {"name": "reserve0", "type": "uint256"},
                    {"name": "reserve1", "type": "uint256"},
                    {"name": "blockTimestampLast", "type": "uint32"}
                ]
            },
            {
                "name": "totalSupply",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "uint256"}]
            },
            {
                "name": "token0",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "address"}]
            },
            {
                "name": "token1",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "address"}]
            },
            {
                "name": "stable",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "bool"}]
            },
            {
                "name": "fees",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "address"}]
            },
            {
                "name": "Swap",
                "type": "event",
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "sender", "type": "address"},
                    {"indexed": False, "name": "amount0In", "type": "uint256"},
                    {"indexed": False, "name": "amount1In", "type": "uint256"},
                    {"indexed": False, "name": "amount0Out", "type": "uint256"},
                    {"indexed": False, "name": "amount1Out", "type": "uint256"},
                    {"indexed": True, "name": "to", "type": "address"}
                ]
            },
            {
                "name": "Sync",
                "type": "event",
                "anonymous": False,
                "inputs": [
                    {"indexed": False, "name": "reserve0", "type": "uint256"},
                    {"indexed": False, "name": "reserve1", "type": "uint256"}
                ]
            },
            {
                "name": "Fees",
                "type": "event",
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "sender", "type": "address"},
                    {"indexed": False, "name": "amount0", "type": "uint256"},
                    {"indexed": False, "name": "amount1", "type": "uint256"}
                ]
            }
        ]
    }
}

# Common Token Addresses on Base
TOKENS = {
    "WETH": "0x4200000000000000000000000000000000000006",
    "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "DAI": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
    "AERO": "0x940181a94A35A4569E4529A3CDfB74e38FD98631",
    "USDbC": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",
}

# Pool Types
POOL_TYPES = {
    "STABLE": True,  # For stablecoin pairs
    "VOLATILE": False,  # For volatile pairs
}

# Fee Tiers (in basis points)
FEE_TIERS = {
    "STABLE": 1,  # 0.01%
    "VOLATILE": 30,  # 0.3%
}

# Slippage Settings
DEFAULT_SLIPPAGE = 0.005  # 0.5%
MAX_SLIPPAGE = 0.03  # 3%

# Gas Settings
GAS_BUFFER = 1.2  # 20% buffer on gas estimates
MAX_GAS_PRICE = 100  # Max 100 gwei