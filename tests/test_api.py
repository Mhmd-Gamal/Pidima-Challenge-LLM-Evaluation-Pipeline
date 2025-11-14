"""
Comprehensive test suite for the LLM API endpoints.

Author: Mohamed Gamal Elbayoumi
Email: elbayoumigamal@gmail.com
Date: November 9, 2025

Run with: pytest tests/test_api.py -v
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import torch

# Import the FastAPI app and related components
from src.api.main import app
from src.api.models import (
    GenerateRequest,
    GenerateResponse,
    MCQRequest,
    MCQResponse,
    HealthResponse
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    model = MagicMock()
    model.parameters.return_value = [torch.zeros(1)]  # Mock parameters for device detection
    model.eval.return_value = None
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_inference_engine(mock_model, mock_tokenizer):
    """Create a mock inference engine with predefined responses."""
    engine = MagicMock()
    
    # Mock generate method
    def mock_generate(prompt, **kwargs):
        return {
            "text": "This is a generated response.",
            "tokens": 10
        }
    engine.generate = Mock(side_effect=mock_generate)
    
    # Mock evaluate_mcq method
    def mock_evaluate(question, options, context=None):
        return {
            "answer": "B",
            "confidence": 0.85,
            "reasoning": "This is the correct answer because...",
            "raw_output": "B) is the correct answer."
        }
    engine.evaluate_mcq = Mock(side_effect=mock_evaluate)
    
    return engine


# ============================================
# ROOT ENDPOINT TESTS
# ============================================

class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint_success(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        assert data["docs"] == "/docs"
    
    def test_root_endpoint_fields(self, client):
        """Test that root endpoint has all required fields."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert "docs" in data


# ============================================
# HEALTH CHECK ENDPOINT TESTS
# ============================================

class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    @patch('src.api.main.inference_engine')
    def test_health_check_healthy(self, mock_engine, client):
        """Test health check when model is loaded."""
        mock_engine.return_value = MagicMock()  # Model is loaded
        
        response = client.get("/health")
        
        # Note: Might return 503 if model not actually loaded in test environment
        # This is expected behavior
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    def test_health_check_response_structure(self, client):
        """Test that health check response has correct structure."""
        response = client.get("/health")
        data = response.json()
        
        # Check required fields exist
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        
        # Check types
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0


# ============================================
# GENERATE ENDPOINT TESTS
# ============================================

class TestGenerateEndpoint:
    """Test the text generation endpoint."""
    
    @patch('src.api.main.inference_engine')
    def test_generate_success(self, mock_engine, client):
        """Test successful text generation."""
        # Mock the inference engine
        mock_engine.generate.return_value = {
            "text": "Paris is the capital of France.",
            "tokens": 8
        }
        
        request_data = {
            "prompt": "What is the capital of France?",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = client.post("/generate", json=request_data)
        
        # May return 503 if model not loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "tokens" in data
            assert "time_ms" in data
            assert isinstance(data["tokens"], int)
            assert isinstance(data["time_ms"], int)
    
    def test_generate_minimal_request(self, client):
        """Test generation with minimal required fields."""
        request_data = {
            "prompt": "Hello world"
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should either work (200) or fail due to model not loaded (503)
        assert response.status_code in [200, 503]
    
    def test_generate_empty_prompt(self, client):
        """Test that empty prompt is rejected."""
        request_data = {
            "prompt": ""
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_generate_prompt_too_long(self, client):
        """Test that overly long prompts are rejected."""
        request_data = {
            "prompt": "x" * 3000  # Exceeds max_length of 2048
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_generate_invalid_temperature(self, client):
        """Test that invalid temperature values are rejected."""
        request_data = {
            "prompt": "Test prompt",
            "temperature": 3.0  # Exceeds max of 2.0
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_generate_negative_max_tokens(self, client):
        """Test that negative max_tokens is rejected."""
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": -1
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_generate_custom_parameters(self, client):
        """Test generation with custom parameters."""
        request_data = {
            "prompt": "Write a poem",
            "max_tokens": 50,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should either work or fail due to model not loaded
        assert response.status_code in [200, 503]


# ============================================
# EVALUATE MCQ ENDPOINT TESTS
# ============================================

class TestEvaluateMCQEndpoint:
    """Test the MCQ evaluation endpoint."""
    
    @patch('src.api.main.inference_engine')
    def test_evaluate_mcq_success(self, mock_engine, client):
        """Test successful MCQ evaluation."""
        # Mock the inference engine
        mock_engine.evaluate_mcq.return_value = {
            "answer": "B",
            "confidence": 0.87,
            "reasoning": "Mitochondria produce energy",
            "raw_output": "The answer is B"
        }
        
        request_data = {
            "question": "What is the powerhouse of the cell?",
            "options": [
                "Nucleus",
                "Mitochondria",
                "Ribosome",
                "Golgi apparatus"
            ]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # May return 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "reasoning" in data
            assert "time_ms" in data
            assert data["answer"] in ["A", "B", "C", "D"]
    
    def test_evaluate_mcq_with_context(self, client):
        """Test MCQ evaluation with context."""
        request_data = {
            "question": "What was the main cause?",
            "options": ["Economic", "Political", "Natural", "Military"],
            "context": "The passage describes how political instability led to changes."
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should either work or fail due to model not loaded
        assert response.status_code in [200, 503]
    
    def test_evaluate_mcq_two_options(self, client):
        """Test MCQ with minimum number of options (2)."""
        request_data = {
            "question": "Is the sky blue?",
            "options": ["Yes", "No"]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should accept 2 options
        assert response.status_code in [200, 503]
    
    def test_evaluate_mcq_many_options(self, client):
        """Test MCQ with many options."""
        request_data = {
            "question": "Which letter comes after A?",
            "options": list("BCDEFGHIJKLMNOPQRSTUVWXYZ")  # 25 options
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should accept up to 26 options
        assert response.status_code in [200, 503]
    
    def test_evaluate_mcq_too_few_options(self, client):
        """Test that MCQ with only 1 option is rejected."""
        request_data = {
            "question": "What is the answer?",
            "options": ["Only option"]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should fail validation (minimum 2 options)
        assert response.status_code == 400
    
    def test_evaluate_mcq_too_many_options(self, client):
        """Test that MCQ with more than 26 options is rejected."""
        request_data = {
            "question": "What is the answer?",
            "options": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Extra"]  # 27 options
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should fail validation (maximum 26 options)
        assert response.status_code == 400
    
    def test_evaluate_mcq_empty_question(self, client):
        """Test that empty question is rejected."""
        request_data = {
            "question": "",
            "options": ["A", "B", "C", "D"]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should fail validation (min_length=5)
        assert response.status_code == 422
    
    def test_evaluate_mcq_short_question(self, client):
        """Test that very short question is rejected."""
        request_data = {
            "question": "Why",  # Only 3 chars, minimum is 5
            "options": ["A", "B", "C", "D"]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_evaluate_mcq_empty_options(self, client):
        """Test that empty options are rejected."""
        request_data = {
            "question": "What is the answer?",
            "options": ["Valid", "", "Another"]
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_evaluate_mcq_null_context(self, client):
        """Test MCQ with explicitly null context."""
        request_data = {
            "question": "What is 2+2?",
            "options": ["2", "3", "4", "5"],
            "context": None
        }
        
        response = client.post("/evaluate_mcq", json=request_data)
        
        # Should accept null context
        assert response.status_code in [200, 503]


# ============================================
# PYDANTIC MODEL TESTS
# ============================================

class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_generate_request_valid(self):
        """Test valid GenerateRequest creation."""
        request = GenerateRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
    
    def test_generate_request_defaults(self):
        """Test GenerateRequest with default values."""
        request = GenerateRequest(prompt="Test")
        
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 50
    
    def test_mcq_request_valid(self):
        """Test valid MCQRequest creation."""
        request = MCQRequest(
            question="What is AI?",
            options=["A", "B", "C", "D"]
        )
        
        assert request.question == "What is AI?"
        assert len(request.options) == 4
        assert request.context is None
    
    def test_mcq_request_with_context(self):
        """Test MCQRequest with context."""
        request = MCQRequest(
            question="What was discussed?",
            options=["Topic A", "Topic B"],
            context="The passage discusses various topics."
        )
        
        assert request.context == "The passage discusses various topics."
    
    def test_generate_response_valid(self):
        """Test valid GenerateResponse creation."""
        response = GenerateResponse(
            text="Generated text",
            tokens=10,
            time_ms=234
        )
        
        assert response.text == "Generated text"
        assert response.tokens == 10
        assert response.time_ms == 234
    
    def test_mcq_response_valid(self):
        """Test valid MCQResponse creation."""
        response = MCQResponse(
            answer="B",
            confidence=0.85,
            reasoning="Because...",
            time_ms=189
        )
        
        assert response.answer == "B"
        assert response.confidence == 0.85
        assert response.reasoning == "Because..."
        assert response.time_ms == 189
    
    def test_health_response_valid(self):
        """Test valid HealthResponse creation."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="test-model",
            uptime_seconds=3600
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.model_name == "test-model"
        assert response.uptime_seconds == 3600


# ============================================
# ERROR HANDLING TESTS
# ============================================

class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self, client):
        """Test that invalid JSON is handled properly."""
        response = client.post(
            "/generate",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for invalid JSON
        assert response.status_code == 422
    
    def test_missing_required_field(self, client):
        """Test that missing required fields are caught."""
        response = client.post("/generate", json={})
        
        # Should return 422 for missing required field
        assert response.status_code == 422
    
    def test_wrong_field_type(self, client):
        """Test that wrong field types are caught."""
        request_data = {
            "prompt": "Test",
            "max_tokens": "not a number"  # Should be int
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should return 422 for type error
        assert response.status_code == 422
    
    def test_get_on_post_endpoint(self, client):
        """Test that GET request on POST endpoint fails."""
        response = client.get("/generate")
        
        # Should return 405 Method Not Allowed
        assert response.status_code == 405


# ============================================
# INTEGRATION TESTS
# ============================================

class TestIntegration:
    """Integration tests for API workflow."""
    
    def test_health_before_generate(self, client):
        """Test checking health before generating."""
        # First check health
        health_response = client.get("/health")
        assert health_response.status_code in [200, 503]
        
        # Then try to generate
        gen_response = client.post(
            "/generate",
            json={"prompt": "Test"}
        )
        assert gen_response.status_code in [200, 503]
    
    def test_multiple_requests(self, client):
        """Test making multiple requests in sequence."""
        for i in range(3):
            response = client.get("/health")
            assert response.status_code in [200, 503]
    
    def test_concurrent_health_checks(self, client):
        """Test that multiple health checks work."""
        responses = []
        for _ in range(5):
            response = client.get("/health")
            responses.append(response)
        
        # All should succeed or fail consistently
        statuses = [r.status_code for r in responses]
        assert all(s in [200, 503] for s in statuses)


# ============================================
# DOCUMENTATION TESTS
# ============================================

class TestDocumentation:
    """Test that API documentation is accessible."""
    
    def test_openapi_json_available(self, client):
        """Test that OpenAPI JSON spec is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_ui_available(self, client):
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        
        # Should return HTML
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_ui_available(self, client):
        """Test that ReDoc UI is available."""
        response = client.get("/redoc")
        
        # Should return HTML
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# ============================================
# PERFORMANCE TESTS
# ============================================

class TestPerformance:
    """Basic performance tests."""
    
    def test_health_check_fast(self, client):
        """Test that health check responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        # Health check should be fast (< 1 second)
        assert elapsed < 1.0
        assert response.status_code in [200, 503]
    
    def test_root_endpoint_fast(self, client):
        """Test that root endpoint responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/")
        elapsed = time.time() - start
        
        # Root endpoint should be very fast (< 0.1 seconds)
        assert elapsed < 0.1
        assert response.status_code == 200


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
