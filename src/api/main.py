"""
FastAPI application for LLM inference and MCQ evaluation.
"""
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    GenerateRequest,
    GenerateResponse,
    MCQRequest,
    MCQResponse,
    HealthResponse
)
from ..llm.loader import ModelLoader
from ..llm.inference import InferenceEngine
from ..utils.config import settings
from ..utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for model and inference engine
model_loader = None
inference_engine = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model_loader, inference_engine
    
    logger.info("Starting LLM API server...")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Device: {settings.DEVICE}")
    
    try:
        # Load model on startup
        model_loader = ModelLoader(
            model_name=settings.MODEL_NAME,
            cache_dir=settings.MODEL_CACHE_DIR,
            device=settings.DEVICE
        )
        
        model, tokenizer = model_loader.load_model()
        inference_engine = InferenceEngine(model=model, tokenizer=tokenizer)
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down LLM API server...")
    if model_loader:
        model_loader.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Pidima LLM Evaluation API",
    description="REST API for LLM inference and MCQ evaluation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Pidima LLM Evaluation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring and orchestration.
    
    Returns:
        HealthResponse with server status and model information
    """
    try:
        model_loaded = inference_engine is not None
        uptime = int(time.time() - start_time)
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_name=settings.MODEL_NAME if model_loaded else None,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text completion from a prompt.
    
    Args:
        request: GenerateRequest with prompt and generation parameters
        
    Returns:
        GenerateResponse with generated text and metadata
        
    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start = time.time()
        
        result = inference_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return GenerateResponse(
            text=result["text"],
            tokens=result["tokens"],
            time_ms=elapsed_ms
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/evaluate_mcq", response_model=MCQResponse, tags=["Evaluation"])
async def evaluate_mcq(request: MCQRequest) -> MCQResponse:
    """
    Evaluate a multiple-choice question and return the predicted answer.
    
    Args:
        request: MCQRequest with question, options, and optional context
        
    Returns:
        MCQResponse with predicted answer, confidence, and reasoning
        
    Raises:
        HTTPException: If model is not loaded or evaluation fails
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Validate options
    if len(request.options) < 2 or len(request.options) > 26:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Options must contain 2-26 items"
        )
    
    try:
        start = time.time()
        
        result = inference_engine.evaluate_mcq(
            question=request.question,
            options=request.options,
            context=request.context
        )
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return MCQResponse(
            answer=result["answer"],
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", ""),
            time_ms=elapsed_ms
        )
        
    except Exception as e:
        logger.error(f"MCQ evaluation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower()
    )