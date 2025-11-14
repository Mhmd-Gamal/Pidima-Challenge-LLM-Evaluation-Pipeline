# Pidima AI Engineer Challenge - LLM Evaluation Pipeline

**Candidate**: Mohamed Gamal Elbayoumi  
**Email**: elbayoumigamal@gmail.com  
**Date**: November 9, 2025

Complete solution for deploying and evaluating an LLM on multiple-choice question benchmarks.

---

## Architecture Overview

- **Model**: Phi-3-mini-4k-instruct (3.8B parameters)
- **Framework**: FastAPI + Transformers
- **Dataset**: MMLU (Massive Multitask Language Understanding)
- **Deployment**: Docker containerized
- **Evaluation**: 150 questions across 10 categories

---

## Quick Start (Choose One)

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate to directory
cd pidima-challenge

# Start services
docker-compose up --build

# Wait for "Model loaded successfully!" message (2-3 minutes)

# Test the API
curl http://localhost:8000/health

# In a new terminal, run evaluation
docker exec -it pidima-llm-api python src/evaluation/run_evaluation.py
```

### Option 2: Local Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start API server
python src/api/main.py

# In new terminal, run evaluation
python src/evaluation/run_evaluation.py
```

---

## Prerequisites

- **Docker & Docker Compose** (for containerized deployment)
- **Python 3.11+** (for local development)
- **8GB+ RAM** (16GB recommended)
- **10GB disk space** (for model weights)

---

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "microsoft/Phi-3-mini-4k-instruct",
  "uptime_seconds": 3600
}
```

### 2. Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```
**Response:**
```json
{
  "text": "The capital of France is Paris...",
  "tokens": 8,
  "time_ms": 234
}
```

### 3. Evaluate MCQ
```bash
curl -X POST http://localhost:8000/evaluate_mcq \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2+2?",
    "options": ["2", "3", "4", "5"],
    "context": null
  }'
```
**Response:**
```json
{
  "answer": "C",
  "confidence": 0.85,
  "reasoning": "Basic arithmetic: 2+2=4",
  "time_ms": 189
}
```

**Interactive Documentation**: Visit http://localhost:8000/docs

---

## Project Structure

```
pidima-challenge/
├── README.md                    # This file
├── COVER_LETTER.md             # Personal approach explanation
├── SUBMISSION_GUIDE.md         # Comprehensive guide
├── Dockerfile                  # Production container
├── docker-compose.yml          # Orchestration
├── requirements.txt            # Dependencies
├── .env.example               # Config template
│
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app (3 endpoints)
│   │   └── models.py         # Pydantic schemas
│   ├── llm/
│   │   ├── loader.py         # Model loading
│   │   └── inference.py      # Inference engine
│   ├── evaluation/
│   │   ├── dataset.py        # MMLU dataset handling
│   │   ├── run_evaluation.py # Evaluation pipeline
│   │   └── metrics.py        # Metrics & analysis
│   └── utils/
│       ├── config.py         # Configuration
│       └── logging.py        # Logging setup
│
├── tests/
│   └── test_api.py           # Unit tests
│
├── docs/
│   └── API_DOCUMENTATION.md  # API reference
│
└── results/                   # Generated outputs
    ├── evaluation_results.json
    ├── error_analysis.md
    ├── metrics.json
    └── visualizations/
        ├── category_accuracy.png
        ├── response_time_dist.png
        ├── confusion_matrix.png
        └── confidence_dist.png
```

---

## Evaluation Pipeline

The pipeline automatically:

1. Downloads MMLU dataset from HuggingFace
2. Samples 150 questions across 10 categories
3. Evaluates each question through the API
4. Generates comprehensive metrics
5. Creates visualizations
6. Produces error analysis report

**Run evaluation:**
```bash
python src/evaluation/run_evaluation.py
```

---

## Key Metrics

The pipeline calculates:

- ✅ Overall accuracy and per-category breakdown
- ✅ Response time statistics (mean, median, std)
- ✅ Confidence score distribution
- ✅ Answer extraction success rate
- ✅ Confusion matrix analysis
- ✅ Error pattern taxonomy

---

## Configuration

Create `.env` file from template:
```bash
cp .env.example .env
```

**Key settings:**
- `MODEL_NAME` - HuggingFace model identifier
- `MODEL_CACHE_DIR` - Where to store model weights
- `DEVICE` - `cpu` or `cuda`
- `MAX_LENGTH` - Maximum generation length
- `EVAL_BATCH_SIZE` - Concurrent API requests (default: 5)

---

## Design Decisions

### 1. Model Selection: Phi-3-mini-4k-instruct
**Why?**
- 3.8B parameters - perfect for 5-8 hour timeline
- Excellent instruction-following capability
- Fast CPU inference (200-300ms per question)
- Well-documented and actively maintained

### 2. Architecture Choices
- **FastAPI**: Async support, auto-generated docs, built-in validation
- **Docker Multi-Stage Build**: Smaller image size, better security
- **Async Evaluation**: 3-4x faster than sequential processing
- **Stratified Sampling**: Balanced category representation

### 3. Answer Extraction Strategy
- 5 different regex patterns for robustness
- Handles varied LLM output formats
- Graceful fallback for extraction failures
- Detailed logging for debugging

---

## Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Overall Accuracy | 65-75% |
| Response Time (mean) | 200-300ms |
| Extraction Success | >95% |
| Questions per minute | 20-30 |
| Memory usage | 4-6 GB |
| Docker image size | 2-3 GB |

**vs Random Baseline**: 25% → ~70% = **2.8x improvement**

---

## Troubleshooting

### Model download is slow
```bash
# Pre-download model
huggingface-cli download microsoft/Phi-3-mini-4k-instruct
```

### Out of memory
```bash
# Reduce batch size in .env
EVAL_BATCH_SIZE=1
```

### API not responding
```bash
# Check logs
docker logs pidima-llm-api

# Restart container
docker-compose restart
```

### Import errors
```bash
# Ensure all __init__.py files exist
find src -type d -exec touch {}/__init__.py \;
```

---

## Testing

Run unit tests:
```bash
pytest tests/test_api.py -v
```

Manual API testing:
```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# Evaluate MCQ
curl -X POST http://localhost:8000/evaluate_mcq \
  -H "Content-Type: application/json" \
  -d '{"question": "Test?", "options": ["A", "B", "C", "D"]}'
```

---

## Production Considerations

✅ Health checks for container orchestration  
✅ Graceful shutdown handling  
✅ Request/response validation  
✅ Structured logging  
✅ Environment-based configuration  
✅ Non-root Docker user for security  
✅ Multi-stage Docker build  
✅ Resource limits (CPU, memory)

---

## Future Enhancements

**Short-term (1-2 days):**
- Few-shot prompting experiments
- Constrained decoding for valid answers
- Enhanced confidence scoring

**Medium-term (1 week):**
- Model comparison (Mistral, Llama)
- LoRA fine-tuning
- API authentication

**Long-term (1 month):**
- RAG integration
- Monitoring dashboard
- Production deployment pipeline

---

## Resources

- [Phi-3 Documentation](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

## License

MIT License - See LICENSE file for details

---

## Author

**Mohamed Gamal Elbayoumi**  
Email: elbayoumigamal@gmail.com  
Challenge: Pidima AI Engineer Technical Interview  
Date: November 9, 2025

---

**Thank you for reviewing this submission!** I'm excited to discuss the implementation details, design decisions, and potential improvements in the follow-up interview.