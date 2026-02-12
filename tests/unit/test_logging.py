"""
Unit Tests for Logging Configuration

Tests the logging configuration to verify proper setup of handlers,
formatters, and log levels across different modules.

Run: pytest tests/unit/test_logging.py -v
"""
import pytest
import logging
import logging.config
import yaml
from pathlib import Path


@pytest.fixture(scope="module")
def config_path():
    """Return the path to logging configuration file."""
    return (
        Path(__file__).parent.parent.parent / 
        "generative_ai_project" / 
        "config" / 
        "logging_config.yaml"
    )


@pytest.fixture(scope="module")
def logging_config(config_path):
    """Load and return logging configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def log_directory():
    """Return the log directory path and ensure it exists."""
    # Il tuo YAML usa "logs/" quindi creiamo la directory relativa alla root del progetto
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture(scope="function")
def configured_logging(config_path, log_directory):
    """Setup logging configuration with log directory created."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
        if 'version' not in config:
            config['version'] = 1
        logging.config.dictConfig(config)
    yield
    # Cleanup
    logging.shutdown()


class TestLoggingConfiguration:
    """Test suite for logging configuration validation."""
    
    def test_config_file_exists(self, config_path):
        """Test that logging configuration file exists."""
        assert config_path.exists(), f"Configuration file not found: {config_path}"
    
    def test_config_file_valid_yaml(self, logging_config):
        """Test that configuration file is valid YAML."""
        assert isinstance(logging_config, dict), "Configuration must be a dictionary"
    
    def test_config_has_required_keys(self, logging_config):
        """Test that configuration contains required keys."""
        required_keys = ['handlers', 'loggers', 'formatters']
        
        for key in required_keys:
            assert key in logging_config, f"Missing required key: {key}"
    
    def test_handlers_defined(self, logging_config):
        """Test that handlers are properly defined."""
        handlers = logging_config.get('handlers', {})
        
        assert len(handlers) > 0, "No handlers defined"
        assert 'console' in handlers, "Console handler not defined"
        # Verifica che esista almeno un file handler
        file_handlers = [k for k in handlers.keys() if 'file' in k.lower()]
        assert len(file_handlers) > 0, "No file handlers defined"
    
    def test_formatters_defined(self, logging_config):
        """Test that formatters are properly defined."""
        formatters = logging_config.get('formatters', {})
        
        assert len(formatters) > 0, "No formatters defined"
    
    def test_loggers_defined(self, logging_config):
        """Test that loggers are properly defined."""
        loggers = logging_config.get('loggers', {})
        
        expected_loggers = ['RansomPipeline', 'LLMClient', 'ErrorHandler']
        
        for logger_name in expected_loggers:
            assert logger_name in loggers, f"Logger '{logger_name}' not defined"


class TestLoggerInitialization:
    """Test suite for logger initialization."""
    
    def test_ransom_pipeline_logger(self, configured_logging):
        """Test RansomPipeline logger initialization."""
        logger = logging.getLogger("RansomPipeline")
        
        assert logger is not None, "RansomPipeline logger is None"
        assert logger.level >= 0, "Invalid log level"
    
    def test_llm_client_logger(self, configured_logging):
        """Test LLMClient logger initialization."""
        logger = logging.getLogger("LLMClient")
        
        assert logger is not None, "LLMClient logger is None"
        assert logger.level >= 0, "Invalid log level"
    
    def test_error_handler_logger(self, configured_logging):
        """Test ErrorHandler logger initialization."""
        logger = logging.getLogger("ErrorHandler")
        
        assert logger is not None, "ErrorHandler logger is None"
        assert logger.level >= 0, "Invalid log level"


class TestLogLevels:
    """Test suite for log level functionality."""
    
    def test_debug_level(self, configured_logging):
        """Test DEBUG level logging."""
        logger = logging.getLogger("RansomPipeline")
        
        try:
            logger.debug("Test debug message")
        except Exception as e:
            pytest.fail(f"DEBUG logging failed: {e}")
    
    def test_info_level(self, configured_logging):
        """Test INFO level logging."""
        logger = logging.getLogger("RansomPipeline")
        
        try:
            logger.info("Test info message")
        except Exception as e:
            pytest.fail(f"INFO logging failed: {e}")
    
    def test_warning_level(self, configured_logging):
        """Test WARNING level logging."""
        logger = logging.getLogger("RansomPipeline")
        
        try:
            logger.warning("Test warning message")
        except Exception as e:
            pytest.fail(f"WARNING logging failed: {e}")
    
    def test_error_level(self, configured_logging):
        """Test ERROR level logging."""
        logger = logging.getLogger("RansomPipeline")
        
        try:
            logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"ERROR logging failed: {e}")
    
    def test_critical_level(self, configured_logging):
        """Test CRITICAL level logging."""
        logger = logging.getLogger("RansomPipeline")
        
        try:
            logger.critical("Test critical message")
        except Exception as e:
            pytest.fail(f"CRITICAL logging failed: {e}")


class TestLogFiles:
    """Test suite for log file creation and management."""
    
    def test_log_directory_structure(self, log_directory):
        """Test that log directory exists."""
        assert log_directory.exists(), "Log directory does not exist"
        assert log_directory.is_dir(), "Log directory is not a directory"
    
    def test_expected_log_files_structure(self, logging_config):
        """Test that configuration defines expected log files."""
        handlers = logging_config.get('handlers', {})
        
        file_handlers = [
            name for name, config in handlers.items() 
            if config.get('class') in [
                'logging.FileHandler',
                'logging.handlers.RotatingFileHandler',
                'logging.handlers.TimedRotatingFileHandler'
            ]
        ]
        
        assert len(file_handlers) > 0, "No file handlers defined in configuration"


class TestLoggingIntegration:
    """Integration tests for logging system."""
    
    def test_multiple_loggers_coexist(self, configured_logging):
        """Test that multiple loggers can coexist and log independently."""
        pipeline_logger = logging.getLogger("RansomPipeline")
        client_logger = logging.getLogger("LLMClient")
        error_logger = logging.getLogger("ErrorHandler")
        
        try:
            pipeline_logger.info("Pipeline message")
            client_logger.warning("Client warning")
            error_logger.error("Error handler message")
        except Exception as e:
            pytest.fail(f"Multiple logger test failed: {e}")
    
    def test_logger_hierarchy(self, configured_logging):
        """Test logger hierarchy and inheritance."""
        parent_logger = logging.getLogger("RansomPipeline")
        child_logger = logging.getLogger("RansomPipeline.SubModule")
        
        assert child_logger is not None, "Child logger not created"
        
        try:
            child_logger.info("Child logger message")
        except Exception as e:
            pytest.fail(f"Logger hierarchy test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
