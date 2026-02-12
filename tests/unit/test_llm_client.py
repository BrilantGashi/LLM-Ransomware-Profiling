"""
Unit Tests for LLMClient Module

Tests API client initialization, response generation, error handling, and configuration.
Run: pytest tests/unit/test_llm_client.py -v
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from generative_ai_project.src.llm.unibs_client import LLMClient


try:
    from openai import OpenAI, APIError, APITimeoutError
except ImportError:
    class OpenAI:
        pass
    class APIError(Exception):
        pass
    class APITimeoutError(Exception):
        pass


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("GPUSTACK_API_KEY", "test_api_key_12345")


@pytest.fixture
def mock_config_file(tmp_path):
    """Create temporary config file for testing"""
    config_content = """
active_model: qwen3

llm_parameters:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  frequency_penalty: 0.1
  presence_penalty: 0.1
"""
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = "Test response content"
    mock_message.reasoning_content = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


class TestLLMClientInitialization:
    """Test LLMClient initialization and configuration loading"""
    
    def test_initialization_with_valid_config(self, mock_env_vars, mock_config_file):
        """Test successful initialization with valid config"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            client = LLMClient(config_path=mock_config_file)
            assert client.model_name == "qwen3"
            assert client.llm_params['temperature'] == 0.7
            assert client.llm_params['max_tokens'] == 2048
    
    def test_initialization_with_model_override(self, mock_env_vars, mock_config_file):
        """Test initialization with model override"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            client = LLMClient(config_path=mock_config_file, model_override="phi4-mini")
            assert client.model_name == "phi4-mini"
    
    def test_missing_api_key_raises_error(self, mock_config_file, monkeypatch):
        """Test that missing API key raises ValueError"""
        monkeypatch.delenv("GPUSTACK_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="GPUSTACK_API_KEY not found"):
            LLMClient(config_path=mock_config_file)
    
    def test_config_file_not_found(self, mock_env_vars, tmp_path):
        """Test handling of missing config file"""
        non_existent_config = str(tmp_path / "nonexistent.yaml")
        
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            with pytest.raises(FileNotFoundError):
                LLMClient(config_path=non_existent_config)
    
    def test_default_parameters_used(self, mock_env_vars, tmp_path):
        """Test that default parameters are used when not in config"""
        minimal_config = tmp_path / "minimal_config.yaml"
        minimal_config.write_text("active_model: qwen3")
        
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            client = LLMClient(config_path=str(minimal_config))
            assert client.model_name == "qwen3"
            assert client.llm_params == {}
    
    def test_openai_client_initialized_correctly(self, mock_env_vars, mock_config_file):
        """Test that OpenAI client is initialized with correct parameters"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            client = LLMClient(config_path=mock_config_file)
            
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs['base_url'] == "https://gpustack.ing.unibs.it/v1"
            assert call_kwargs['api_key'] == "test_api_key_12345"
            assert call_kwargs['timeout'] == 480.0


class TestGenerateResponse:
    """Test response generation functionality"""
    
    def test_successful_response_generation(self, mock_env_vars, mock_config_file):
        """Test successful API call and response extraction"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Test response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            
            messages = [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            result = client.generate_response(messages)
            
            assert result['content'] == "Test response"
            assert 'reasoning' not in result
    
    def test_response_with_reasoning(self, mock_env_vars, mock_config_file):
        """Test response extraction with reasoning content"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Answer content"
            mock_message.reasoning_content = "Reasoning steps"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            result = client.generate_response(messages)
            
            assert result['content'] == "Answer content"
            assert result['reasoning'] == "Reasoning steps"
    
    def test_api_parameters_passed_correctly(self, mock_env_vars, mock_config_file):
        """Test that LLM parameters are passed to API call"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            client.generate_response(messages)
            
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs['model'] == "qwen3"
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['top_p'] == 0.9
            assert call_kwargs['max_tokens'] == 2048
            assert call_kwargs['frequency_penalty'] == 0.1
            assert call_kwargs['presence_penalty'] == 0.1
    
    def test_empty_messages_handled(self, mock_env_vars, mock_config_file):
        """Test handling of empty message list"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception("Empty messages")
            
            client = LLMClient(config_path=mock_config_file)
            
            with pytest.raises(RuntimeError, match="API call failed"):
                client.generate_response([])
    
    def test_multiple_messages_in_conversation(self, mock_env_vars, mock_config_file):
        """Test handling of multi-turn conversation"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"}
            ]
            
            result = client.generate_response(messages)
            
            assert 'content' in result
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert len(call_kwargs['messages']) == 4


class TestErrorHandling:
    """Test error handling in API calls"""
    
    def test_api_timeout_error_raised(self, mock_env_vars, mock_config_file):
        """Test that timeout errors are properly raised"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = APITimeoutError("Timeout")
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(RuntimeError, match="API call failed"):
                client.generate_response(messages)

    def test_api_error_raised(self, mock_env_vars, mock_config_file):
        """Test that API errors are properly raised"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            # Crea mock completo per APIError con tutti i parametri richiesti
            mock_request = MagicMock()
            mock_body = MagicMock()
            mock_client.chat.completions.create.side_effect = APIError(
                "Invalid key", 
                request=mock_request,
                body=mock_body
            )
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(RuntimeError, match="API call failed"):
                client.generate_response(messages)

    
    def test_generic_exception_handling(self, mock_env_vars, mock_config_file):
        """Test handling of unexpected exceptions"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception("Unexpected error")
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(RuntimeError, match="API call failed"):
                client.generate_response(messages)
    
    def test_error_message_includes_model_name(self, mock_env_vars, mock_config_file):
        """Test that error messages include the model name"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception("Test error")
            
            client = LLMClient(config_path=mock_config_file, model_override="test-model")
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(RuntimeError, match="test-model"):
                client.generate_response(messages)


class TestModelConfiguration:
    """Test model configuration and parameter handling"""
    
    def test_custom_temperature(self, mock_env_vars, tmp_path):
        """Test custom temperature parameter"""
        config_content = """
active_model: qwen3
llm_parameters:
  temperature: 0.3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            client = LLMClient(config_path=str(config_file))
            assert client.llm_params['temperature'] == 0.3
    
    def test_missing_parameters_use_defaults(self, mock_env_vars, mock_config_file):
        """Test that missing parameters use default values in API call"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            minimal_config = """
active_model: qwen3
llm_parameters: {}
"""
            config_file = Path(mock_config_file).parent / "minimal.yaml"
            config_file.write_text(minimal_config)
            
            client = LLMClient(config_path=str(config_file))
            messages = [{"role": "user", "content": "Test"}]
            client.generate_response(messages)
            
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs['temperature'] == 0.6
            assert call_kwargs['max_tokens'] == 1024
    
    def test_all_llm_parameters_configurable(self, mock_env_vars, tmp_path):
        """Test that all LLM parameters can be configured"""
        config_content = """
active_model: qwen3
llm_parameters:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 4096
  frequency_penalty: 0.5
  presence_penalty: 0.5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI'):
            client = LLMClient(config_path=str(config_file))
            
            assert client.llm_params['temperature'] == 0.8
            assert client.llm_params['top_p'] == 0.95
            assert client.llm_params['max_tokens'] == 4096
            assert client.llm_params['frequency_penalty'] == 0.5
            assert client.llm_params['presence_penalty'] == 0.5


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_very_long_message(self, mock_env_vars, mock_config_file):
        """Test handling of very long messages"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            
            long_content = "x" * 10000
            messages = [{"role": "user", "content": long_content}]
            
            result = client.generate_response(messages)
            assert 'content' in result
    
    def test_special_characters_in_messages(self, mock_env_vars, mock_config_file):
        """Test handling of special characters"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Response"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            
            messages = [{"role": "user", "content": "Test with special chars @#$%"}]
            
            result = client.generate_response(messages)
            assert 'content' in result
    
    def test_empty_response_content(self, mock_env_vars, mock_config_file):
        """Test handling of empty response content"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = ""
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            result = client.generate_response(messages)
            assert result['content'] == ""
    
    def test_unicode_content(self, mock_env_vars, mock_config_file):
        """Test handling of Unicode characters"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Hello world"
            mock_message.reasoning_content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient(config_path=mock_config_file)
            messages = [{"role": "user", "content": "Test"}]
            
            result = client.generate_response(messages)
            assert "Hello" in result['content']


class TestIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_typical_conversation_flow(self, mock_env_vars, mock_config_file):
        """Test a typical multi-turn conversation"""
        with patch('generative_ai_project.src.llm.unibs_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            
            responses = ["Response 1", "Response 2", "Response 3"]
            call_count = {'count': 0}
            
            def create_mock_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                
                mock_message.content = responses[call_count['count']]
                mock_message.reasoning_content = None
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                
                call_count['count'] += 1
                return mock_response
            
            mock_client.chat.completions.create.side_effect = create_mock_response
            
            client = LLMClient(config_path=mock_config_file)
            
            conversation = [
                {"role": "system", "content": "You are helpful."}
            ]
            
            conversation.append({"role": "user", "content": "Hello"})
            result1 = client.generate_response(conversation)
            assert result1['content'] == "Response 1"
            
            conversation.append({"role": "assistant", "content": result1['content']})
            conversation.append({"role": "user", "content": "Follow up"})
            result2 = client.generate_response(conversation)
            assert result2['content'] == "Response 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
