# New Project Template Using GCP/CDP Setup

## Minimal Project Structure

```
my-defi-project/
├── src/
│   ├── __init__.py
│   ├── cdp/                    # Copy from athena-ai
│   ├── blockchain/             # Copy from athena-ai
│   ├── gcp/                    # Copy from athena-ai
│   └── main.py                 # Your application logic
├── config/
│   ├── __init__.py
│   ├── settings.py             # Copy from athena-ai
│   └── contracts.py            # Modify for your contracts
├── scripts/
│   ├── setup_secrets.py        # Copy from athena-ai
│   └── test_setup.py           # New - test your setup
├── .env.example                # Copy from athena-ai
├── .gitignore                  # Copy from athena-ai
├── requirements.txt            # Copy and modify
├── Dockerfile                  # Copy and modify
└── README.md                   # Your project docs
```

## Example main.py for New Project

```python
"""
Example DeFi application using GCP/CDP setup from Athena
"""
import asyncio
import logging
from decimal import Decimal

from src.cdp.base_client import BaseClient
from src.gcp.firestore_client import FirestoreClient
from src.blockchain.rpc_reader import RPCReader
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDeFiBot:
    """Example DeFi bot using the GCP/CDP infrastructure."""
    
    def __init__(self):
        self.base_client = BaseClient()
        self.firestore = FirestoreClient()
        self.running = False
        
    async def initialize(self):
        """Initialize all components."""
        # Initialize CDP client
        await self.base_client.initialize()
        logger.info(f"Bot wallet: {self.base_client.get_wallet_address()}")
        
        # Initialize Firestore
        await self.firestore.initialize()
        logger.info("Firestore connected")
        
        # Check balances
        usdc_balance = await self.base_client.get_balance("USDC")
        eth_balance = await self.base_client.get_balance("ETH")
        logger.info(f"Balances - USDC: {usdc_balance}, ETH: {eth_balance}")
        
    async def monitor_pools(self):
        """Example: Monitor DEX pools."""
        while self.running:
            try:
                # Get pool info using CDP client
                pool_info = await self.base_client.get_pool_info("WETH", "USDC", False)
                
                if pool_info:
                    logger.info(f"Pool TVL: ${pool_info.get('tvl', 0):,.2f}")
                    
                    # Store in Firestore
                    await self.firestore.set_document(
                        "pool_snapshots",
                        f"weth_usdc_{int(asyncio.get_event_loop().time())}",
                        {
                            "tvl": float(pool_info.get("tvl", 0)),
                            "reserves": {
                                "weth": float(pool_info.get("reserve0", 0)),
                                "usdc": float(pool_info.get("reserve1", 0))
                            },
                            "timestamp": self.firestore.get_timestamp()
                        }
                    )
                    
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
                
    async def run(self):
        """Run the bot."""
        await self.initialize()
        
        self.running = True
        logger.info("Starting DeFi bot...")
        
        # Start monitoring
        await self.monitor_pools()
        
    async def stop(self):
        """Stop the bot."""
        self.running = False
        logger.info("Bot stopped")


async def main():
    """Main entry point."""
    bot = MyDeFiBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Script (scripts/test_setup.py)

```python
#!/usr/bin/env python3
"""
Test script to verify GCP/CDP setup is working correctly.
"""
import asyncio
import logging
from decimal import Decimal

from src.cdp.base_client import BaseClient
from src.gcp.firestore_client import FirestoreClient
from src.gcp.secret_manager import get_secret
from src.blockchain.rpc_reader import RPCReader
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_setup():
    """Test all components are working."""
    logger.info("=== Testing GCP/CDP Setup ===")
    
    # Test 1: Settings and secrets
    logger.info("\n1. Testing settings and secrets...")
    try:
        # Check GCP project
        logger.info(f"✓ GCP Project: {settings.gcp_project_id}")
        
        # Check if secrets are accessible
        test_secret = get_secret("cdp-api-key")
        if test_secret:
            logger.info("✓ Secret Manager is working")
        else:
            logger.warning("✗ Could not access secrets")
    except Exception as e:
        logger.error(f"✗ Settings error: {e}")
        
    # Test 2: CDP Client
    logger.info("\n2. Testing CDP client...")
    try:
        client = BaseClient()
        await client.initialize()
        wallet_address = client.get_wallet_address()
        logger.info(f"✓ CDP wallet initialized: {wallet_address}")
        
        # Check balance
        eth_balance = await client.get_balance("ETH")
        logger.info(f"✓ ETH balance: {eth_balance}")
    except Exception as e:
        logger.error(f"✗ CDP error: {e}")
        
    # Test 3: Firestore
    logger.info("\n3. Testing Firestore...")
    try:
        firestore = FirestoreClient()
        await firestore.initialize()
        
        # Write test document
        await firestore.set_document(
            "test_collection",
            "test_doc",
            {"test": True, "timestamp": firestore.get_timestamp()}
        )
        
        # Read it back
        doc = await firestore.get_document("test_collection", "test_doc")
        if doc and doc.get("test"):
            logger.info("✓ Firestore read/write working")
        else:
            logger.error("✗ Firestore read failed")
    except Exception as e:
        logger.error(f"✗ Firestore error: {e}")
        
    # Test 4: RPC Reader
    logger.info("\n4. Testing RPC reader...")
    try:
        async with RPCReader(settings.cdp_rpc_url) as reader:
            block = await reader._get_current_block()
            if block:
                logger.info(f"✓ RPC reader working - current block: {block}")
            else:
                logger.error("✗ Could not get current block")
    except Exception as e:
        logger.error(f"✗ RPC error: {e}")
        
    logger.info("\n=== Setup Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_setup())
```

## Quick Copy Commands

```bash
# Set variables
ATHENA_DIR="/Users/abhisonu/Projects/project Athena/athena-ai"
NEW_PROJECT_DIR="/path/to/new/project"

# Create directory structure
mkdir -p $NEW_PROJECT_DIR/{src,config,scripts}

# Copy core components
cp -r $ATHENA_DIR/src/cdp $NEW_PROJECT_DIR/src/
cp -r $ATHENA_DIR/src/blockchain $NEW_PROJECT_DIR/src/
cp -r $ATHENA_DIR/src/gcp $NEW_PROJECT_DIR/src/
cp -r $ATHENA_DIR/config/* $NEW_PROJECT_DIR/config/
cp $ATHENA_DIR/scripts/setup_secrets.py $NEW_PROJECT_DIR/scripts/
cp $ATHENA_DIR/scripts/update_cdp_config.py $NEW_PROJECT_DIR/scripts/

# Copy configuration files
cp $ATHENA_DIR/.env.example $NEW_PROJECT_DIR/
cp $ATHENA_DIR/.gitignore $NEW_PROJECT_DIR/
cp $ATHENA_DIR/requirements.txt $NEW_PROJECT_DIR/
cp $ATHENA_DIR/Dockerfile $NEW_PROJECT_DIR/

# Create __init__.py files
touch $NEW_PROJECT_DIR/src/__init__.py
touch $NEW_PROJECT_DIR/config/__init__.py
touch $NEW_PROJECT_DIR/scripts/__init__.py
```

## Customization Checklist

After copying the files, customize for your project:

1. **Update config/contracts.py**
   - Replace Aerodrome contracts with your target protocol
   - Update token addresses for your needs

2. **Modify config/settings.py**
   - Remove Athena-specific settings
   - Add your project-specific configurations

3. **Update requirements.txt**
   - Remove unnecessary dependencies
   - Add your project-specific packages

4. **Create your main application**
   - Use the example main.py as a starting point
   - Implement your DeFi strategy

5. **Update .env.example**
   - Remove Athena-specific variables
   - Add your project variables

## Deployment Checklist

1. **GCP Project Setup**
   - [ ] Create new GCP project
   - [ ] Enable required APIs
   - [ ] Create service account
   - [ ] Initialize Firestore

2. **CDP Setup**
   - [ ] Get CDP API credentials
   - [ ] Get CDP Client API key
   - [ ] Store in Secret Manager

3. **Local Testing**
   - [ ] Run test_setup.py
   - [ ] Verify all components work
   - [ ] Test with small amounts

4. **Production Deployment**
   - [ ] Deploy to Cloud Run
   - [ ] Set up monitoring
   - [ ] Configure alerts