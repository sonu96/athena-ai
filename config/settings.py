"""
Athena AI Configuration Settings
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with Google Secret Manager integration."""
    
    # GCP Configuration (needed first for Secret Manager)
    gcp_project_id: str = Field(..., env="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-central1", env="GCP_REGION")
    firestore_database: str = Field(default="(default)", env="FIRESTORE_DATABASE")
    
    # CDP Configuration - will be loaded from Secret Manager
    cdp_api_key: Optional[str] = Field(None, env="CDP_API_KEY_ID")
    cdp_api_secret: Optional[str] = Field(None, env="CDP_API_KEY_SECRET")
    cdp_wallet_secret: Optional[str] = Field(None, env="CDP_WALLET_SECRET")
    
    @validator("cdp_api_key", pre=False, always=True)
    def load_cdp_api_key(cls, v, values):
        if v:
            return v
        # Load from Secret Manager if not in env
        try:
            from src.gcp.secret_manager import get_secret
            return get_secret("cdp-api-key", "CDP_API_KEY_ID")
        except Exception as e:
            # If we can't load from Secret Manager, validation will fail
            raise ValueError(f"Could not load cdp_api_key from Secret Manager: {e}")
    
    @validator("cdp_api_secret", pre=False, always=True)
    def load_cdp_api_secret(cls, v, values):
        if v:
            return v
        # Load from Secret Manager if not in env
        try:
            from src.gcp.secret_manager import get_secret
            return get_secret("cdp-api-secret", "CDP_API_KEY_SECRET")
        except Exception as e:
            # If we can't load from Secret Manager, validation will fail
            raise ValueError(f"Could not load cdp_api_secret from Secret Manager: {e}")
    
    # Google AI Configuration
    google_ai_model: str = Field(default="gemini-1.5-flash", env="GOOGLE_AI_MODEL")
    google_location: str = Field(default="us-central1", env="GOOGLE_LOCATION")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    
    @validator("google_api_key", pre=False, always=True)
    def load_google_api_key(cls, v, values):
        """Load Google API key from Secret Manager if not in environment."""
        if v is not None:
            return v
        if os.getenv("GOOGLE_API_KEY"):
            return os.getenv("GOOGLE_API_KEY")
        # Try to get from Secret Manager
        try:
            from src.gcp.secret_manager import get_secret
            return get_secret("google-api-key", "GOOGLE_API_KEY")
        except Exception:
            # Google API key is optional for now
            return None
    
    @property
    def google_cloud_project(self) -> str:
        """Return GCP project ID for Google AI."""
        return self.gcp_project_id
    
    # LangSmith Configuration - will be loaded from Secret Manager
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="athena-ai", env="LANGSMITH_PROJECT")
    
    @validator("langsmith_api_key", pre=False, always=True)
    def load_langsmith_api_key(cls, v, values):
        if v:
            return v
        # Load from Secret Manager if not in env
        if "gcp_project_id" in values:
            try:
                from src.gcp.secret_manager import get_secret
                return get_secret("langsmith-api-key", "LANGSMITH_API_KEY")
            except:
                return None  # LangSmith is optional
        return v
    
    # Base Chain Configuration
    base_rpc_url: str = Field(..., env="BASE_RPC_URL")
    chain_id: int = Field(default=8453, env="CHAIN_ID")
    
    # Memory Configuration - will be loaded from Secret Manager
    mem0_api_key: Optional[str] = Field(None, env="MEM0_API_KEY")
    
    @validator("mem0_api_key", pre=False, always=True)
    def load_mem0_api_key(cls, v, values):
        if v:
            return v
        # Load from Secret Manager if not in env
        if "gcp_project_id" in values:
            try:
                from src.gcp.secret_manager import get_secret
                return get_secret("mem0-api-key", "MEM0_API_KEY")
            except:
                return None  # Mem0 can work without API key
        return v
    
    @validator("cdp_wallet_secret", pre=False, always=True)
    def load_cdp_wallet_secret(cls, v, values):
        if v:
            return v
        # Load from Secret Manager if not in env
        if "gcp_project_id" in values:
            try:
                from src.gcp.secret_manager import get_secret
                return get_secret("cdp-wallet-secret", "CDP_WALLET_SECRET")
            except:
                return None  # Wallet secret is optional - will be generated if needed
        return v
    
    # Agent Configuration
    agent_wallet_id: Optional[str] = Field(None, env="AGENT_WALLET_ID")
    agent_cycle_time: int = Field(default=300, env="AGENT_CYCLE_TIME")  # seconds
    agent_max_position_size: float = Field(default=1000.0, env="AGENT_MAX_POSITION_SIZE")  # USD
    agent_risk_limit: float = Field(default=0.02, env="AGENT_RISK_LIMIT")  # 2% max loss
    
    # API Configuration
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Aerodrome Contract Addresses
    aerodrome_router: str = Field(
        default="0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
        env="AERODROME_ROUTER"
    )
    aerodrome_factory: str = Field(
        default="0x420DD381b31aEf6683db6B902084cB0FFECe40Da",
        env="AERODROME_FACTORY"
    )
    aerodrome_voter: str = Field(
        default="0x16613524e02ad97eDfeF371bC883F2F5d6C480A5",
        env="AERODROME_VOTER"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Strategy Configuration
STRATEGIES = {
    "liquidity_provision": {
        "enabled": True,
        "min_apr": 20.0,  # Minimum 20% APR
        "max_il_tolerance": 0.05,  # Max 5% IL tolerance
    },
    "arbitrage": {
        "enabled": True,
        "min_profit": 10.0,  # Minimum $10 profit
        "max_gas_percent": 0.3,  # Max 30% of profit for gas
    },
    "yield_farming": {
        "enabled": True,
        "compound_frequency": 14400,  # Every 4 hours
        "min_pending_rewards": 5.0,  # Minimum $5 to claim
    },
    "vote_optimization": {
        "enabled": True,
        "min_bribe_apr": 50.0,  # Minimum 50% APR from bribes
    },
}


# Memory Categories
MEMORY_CATEGORIES = [
    "market_pattern",
    "gas_optimization",
    "strategy_performance",
    "pool_behavior",
    "user_preference",
    "error_learning",
    "profit_source",
]


# Agent Emotional States
EMOTIONAL_STATES = {
    "confident": {"threshold": 0.8, "description": "High success rate recently"},
    "cautious": {"threshold": 0.5, "description": "Mixed results, being careful"},
    "curious": {"threshold": 0.3, "description": "Exploring new strategies"},
    "learning": {"threshold": 0.0, "description": "Gathering data"},
}