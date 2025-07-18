"""
CDP SDK Version Check
"""
import sys
import logging

logger = logging.getLogger(__name__)

def check_cdp_version():
    """Check that we're using the correct CDP SDK version."""
    try:
        import cdp
        version = getattr(cdp, '__version__', 'unknown')
        
        # Parse version
        if version != 'unknown':
            major, minor, patch = map(int, version.split('.'))
            
            # Require at least version 1.23.0
            if major < 1 or (major == 1 and minor < 23):
                logger.error(f"CDP SDK version {version} is too old. Please upgrade to 1.23.0 or later.")
                logger.error("Run: pip install --upgrade cdp-sdk")
                sys.exit(1)
            else:
                logger.info(f"✅ CDP SDK version {version} is compatible")
        else:
            logger.warning("Could not determine CDP SDK version")
            
    except ImportError:
        logger.error("CDP SDK not installed. Run: pip install cdp-sdk")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Error checking CDP SDK version: {e}")

# Run check on import
check_cdp_version()