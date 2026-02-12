"""
Unit Tests for EmbeddingClient Module

Tests embedding generation, error handling, and API integration.
Run: pytest tests/unit/test_embedding_client.py -v
"""
import pytest
import os
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generative_ai_project.src.llm.embedding_client import EmbeddingClient


class TestEmbeddingClientInitialization:
    """Test EmbeddingClient initialization and configuration"""
    
    @patch.dict(os.environ, {
        'GPUSTACK_API_KEY': 'test-key-123',
        'GPUSTACK_BASE_URL': 'https://test.api.com/v1'
    })
    def test_initialization_with_default_model(self):
        """Test initialization with default model"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI'):
            client = EmbeddingClient()
            assert client.model == "nomic-embed-text-v1.5"
            assert client.api_key == "test-key-123"
            assert client.base_url == "https://test.api.com/v1"
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_initialization_with_custom_model(self):
        """Test initialization with custom model"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI'):
            client = EmbeddingClient(model="qwen3-embedding")
            assert client.model == "qwen3-embedding"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="GPUSTACK_API_KEY environment variable not configured"):
            EmbeddingClient()
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_default_base_url_used(self):
        """Test that default base URL is used when not in env"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI'):
            client = EmbeddingClient()
            assert client.base_url == "https://gpustack.ing.unibs.it/v1"
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_unsupported_model_logs_warning(self, caplog):
        """Test that unsupported model logs warning"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI'):
            client = EmbeddingClient(model="unsupported-model")
            assert "not in supported list" in caplog.text
            assert client.model == "unsupported-model"
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_openai_client_initialized(self):
        """Test that OpenAI client is initialized correctly"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            client = EmbeddingClient()
            mock_openai.assert_called_once_with(
                base_url=client.base_url,
                api_key=client.api_key
            )


class TestEmbedTexts:
    """Test embedding generation functionality"""
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_successful_single_text_embedding(self):
        """Test successful embedding generation for single text"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            embeddings = client.embed_texts(["Hello world"])
            
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            
            mock_client.embeddings.create.assert_called_once_with(
                model="nomic-embed-text-v1.5",
                input=["Hello world"]
            )
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_successful_multiple_texts_embedding(self):
        """Test successful embedding generation for multiple texts"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6]),
                MagicMock(embedding=[0.7, 0.8, 0.9])
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            texts = ["Text 1", "Text 2", "Text 3"]
            embeddings = client.embed_texts(texts)
            
            assert len(embeddings) == 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            assert embeddings[2] == [0.7, 0.8, 0.9]
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_embedding_dimensions_consistent(self):
        """Test that all embeddings have same dimensions"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            embedding_dim = 768
            mock_response.data = [
                MagicMock(embedding=[0.1] * embedding_dim),
                MagicMock(embedding=[0.2] * embedding_dim)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            embeddings = client.embed_texts(["Text 1", "Text 2"])
            
            assert all(len(emb) == embedding_dim for emb in embeddings)
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_empty_text_list_raises_error(self):
        """Test that empty text list causes IndexError (expected behavior)"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = []
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            
            # The current implementation raises IndexError for empty lists
            with pytest.raises(IndexError):
                client.embed_texts([])
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_model_parameter_passed_correctly(self):
        """Test that custom model is passed to API"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(model="qwen3-embedding")
            client.embed_texts(["Test"])
            
            mock_client.embeddings.create.assert_called_with(
                model="qwen3-embedding",
                input=["Test"]
            )
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_logging_on_successful_embedding(self, caplog):
        """Test that successful embedding logs info"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            with caplog.at_level(logging.INFO, logger='EmbeddingClient'):
                client = EmbeddingClient()
                client.embed_texts(["Text 1", "Text 2"])
            
            assert "Generated 2 embeddings" in caplog.text
            assert "dim=3" in caplog.text


class TestErrorHandling:
    """Test error handling in embedding generation"""
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_api_error_raised_and_logged(self, caplog):
        """Test that API errors are raised and logged"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.embeddings.create.side_effect = Exception("API Error: Connection failed")
            
            client = EmbeddingClient()
            
            with pytest.raises(Exception, match="API Error: Connection failed"):
                client.embed_texts(["Test"])
            
            assert "Embedding error" in caplog.text
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_network_error_handled(self):
        """Test that network errors are handled"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.embeddings.create.side_effect = ConnectionError("Network unreachable")
            
            client = EmbeddingClient()
            
            with pytest.raises(ConnectionError):
                client.embed_texts(["Test"])
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_invalid_response_format_handled(self):
        """Test handling of invalid API response format"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = None
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            
            with pytest.raises(Exception):
                client.embed_texts(["Test"])


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_very_long_text(self):
        """Test embedding generation for very long text"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 768)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            long_text = "word " * 10000  # Very long text
            embeddings = client.embed_texts([long_text])
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 768
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_special_characters_in_text(self):
        """Test embedding with special characters"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            special_text = "Hello! @#$%^&*() 你好 مرحبا"
            embeddings = client.embed_texts([special_text])
            
            assert len(embeddings) == 1
            
            mock_client.embeddings.create.assert_called_with(
                model="nomic-embed-text-v1.5",
                input=[special_text]
            )
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_unicode_content(self):
        """Test embedding with unicode characters"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4])
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            texts = ["Hello 世界", "Emoji 🚀🎉"]
            embeddings = client.embed_texts(texts)
            
            assert len(embeddings) == 2
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_whitespace_only_text(self):
        """Test embedding with whitespace-only text"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.0, 0.0, 0.0])]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            embeddings = client.embed_texts(["   \n\t  "])
            
            assert len(embeddings) == 1
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_large_batch_of_texts(self):
        """Test embedding generation for large batch"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            num_texts = 100
            mock_response.data = [
                MagicMock(embedding=[float(i)] * 10)
                for i in range(num_texts)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            texts = [f"Text {i}" for i in range(num_texts)]
            embeddings = client.embed_texts(texts)
            
            assert len(embeddings) == num_texts
            assert all(len(emb) == 10 for emb in embeddings)
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_newlines_and_tabs_in_text(self):
        """Test embedding with newlines and tabs"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient()
            text_with_whitespace = "Line 1\nLine 2\tTabbed\r\nWindows line"
            embeddings = client.embed_texts([text_with_whitespace])
            
            assert len(embeddings) == 1


class TestModelSupport:
    """Test support for different embedding models"""
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_nomic_embed_model(self):
        """Test nomic-embed-text-v1.5 model"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 768)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(model="nomic-embed-text-v1.5")
            embeddings = client.embed_texts(["Test"])
            
            assert len(embeddings[0]) == 768
    
    @patch.dict(os.environ, {'GPUSTACK_API_KEY': 'test-key-123'})
    def test_qwen3_embedding_model(self):
        """Test qwen3-embedding model"""
        with patch('generative_ai_project.src.llm.embedding_client.OpenAI') as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(model="qwen3-embedding")
            embeddings = client.embed_texts(["Test"])
            
            mock_client.embeddings.create.assert_called_with(
                model="qwen3-embedding",
                input=["Test"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
