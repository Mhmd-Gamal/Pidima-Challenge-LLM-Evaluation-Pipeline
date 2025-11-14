"""
Pydantic models for API request and response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class GenerateRequest(BaseModel):
    """Request model for text generation endpoint."""
    
    prompt: str = Field(
        ...,
        description="Input prompt for text generation",
        min_length=1,
        max_length=2048
    )
    max_tokens: int = Field(
        default=100,
        description="Maximum number of tokens to generate",
        ge=1,
        le=2048
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0
    )
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling threshold",
        ge=0.0,
        le=1.0
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling parameter",
        ge=0,
        le=100
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the capital of France?",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        }


class GenerateResponse(BaseModel):
    """Response model for text generation endpoint."""
    
    text: str = Field(..., description="Generated text completion")
    tokens: int = Field(..., description="Number of tokens generated")
    time_ms: int = Field(..., description="Generation time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The capital of France is Paris.",
                "tokens": 8,
                "time_ms": 234
            }
        }


class MCQRequest(BaseModel):
    """Request model for MCQ evaluation endpoint."""
    
    question: str = Field(
        ...,
        description="The multiple-choice question text",
        min_length=5,
        max_length=2048
    )
    options: List[str] = Field(
        ...,
        description="List of answer options",
        min_length=2,
        max_length=26
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context or passage for the question",
        max_length=4096
    )
    
    @validator('options')
    def validate_options(cls, v):
        """Ensure all options are non-empty strings."""
        if not all(isinstance(opt, str) and opt.strip() for opt in v):
            raise ValueError("All options must be non-empty strings")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the primary function of mitochondria?",
                "options": [
                    "Protein synthesis",
                    "Energy production",
                    "DNA replication",
                    "Cell division"
                ],
                "context": None
            }
        }


class MCQResponse(BaseModel):
    """Response model for MCQ evaluation endpoint."""
    
    answer: str = Field(
        ...,
        description="Predicted answer (A, B, C, D, etc.)"
    )
    confidence: float = Field(
        default=0.0,
        description="Model confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the answer"
    )
    time_ms: int = Field(
        ...,
        description="Evaluation time in milliseconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "B",
                "confidence": 0.87,
                "reasoning": "Mitochondria are known as the powerhouse of the cell",
                "time_ms": 189
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the loaded model"
    )
    uptime_seconds: int = Field(
        ...,
        description="Server uptime in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "uptime_seconds": 3600
            }
        }