"""
Test script for logging configuration.
Run from project root: python -m tests.test_logging
"""
import logging
import logging.config
import yaml
from pathlib import Path

# Load config (adjust path since we're in tests/)
config_path = Path(__file__).parent.parent / "generative_ai_project" / "config" / "logging_config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

# Test different loggers
pipeline_logger = logging.getLogger("RansomPipeline")
client_logger = logging.getLogger("UniBS_Client")
error_logger = logging.getLogger("ErrorHandler")

# Test log levels
print("🧪 Testing logging configuration...\n")

pipeline_logger.debug("Debug message (file only)")
pipeline_logger.info("Info message (file only)")
pipeline_logger.warning("Warning message (console + file)")
pipeline_logger.error("Error message (console + file + errors.log)")

client_logger.warning("API timeout simulation (api_errors.log)")
error_logger.error("Critical error simulation")

print("\n✅ Check generative_ai_project/logs/ directory for output files:")
print("   - project.log (all logs)")
print("   - errors.log (errors only)")
print("   - api_errors.log (API issues)")
