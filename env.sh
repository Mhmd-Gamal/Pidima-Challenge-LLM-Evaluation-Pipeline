# Pidima LLM Evaluation Pipeline Configuration
# Author: Mohamed Gamal Elbayoumi
# Date: 2025-11-09

# ============================================
# MODEL CONFIGURATION
# ============================================

# HuggingFace model identifier
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct

# Directory to cache downloaded model weights
MODEL_CACHE_DIR=./models

# Device to run inference on: 'cpu' or 'cuda'
# Note: Use 'cuda' only if you have a compatible GPU
DEVICE=cpu

# Enable 4-bit quantization (GPU only, reduces memory usage)
USE_4BIT=false

# ============================================
# GENERATION SETTINGS
# ============================================

# Maximum sequence length for model
MAX_LENGTH=512

# Default max tokens for text generation
DEFAULT_MAX_TOKENS=100

# ============================================
# API SETTINGS
# ============================================

# API server host (0.0.0.0 for all interfaces)
API_HOST=0.0.0.0

# API server port
API_PORT=8000

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# ============================================
# EVALUATION SETTINGS
# ============================================

# Temperature for evaluation (0.0 for deterministic results)
EVAL_TEMPERATURE=0.0

# Number of concurrent API requests during evaluation
# Higher = faster but more memory usage
EVAL_BATCH_SIZE=5

# ============================================
# NOTES
# ============================================
# 
# For GPU inference:
# - Set DEVICE=cuda
# - Set USE_4BIT=true (optional, saves memory)
# - Ensure you have PyTorch with CUDA support installed
#
# For faster evaluation:
# - Increase EVAL_BATCH_SIZE (max recommended: 10)
#
# For lower memory usage:
# - Decrease EVAL_BATCH_SIZE to 1-2
# - Use 4-bit quantization if on GPU
