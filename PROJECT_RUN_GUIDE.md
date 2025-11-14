# Pidima LLM Evaluation Project - Setup Complete

## Project Status: READY TO RUN

The Pidima LLM Evaluation Pipeline project has been successfully set up and is ready for execution.

###Installed Components:
- ✓ Virtual environment created (`venv`)
- ✓ All Python dependencies installed (FastAPI, PyTorch, Transformers, etc.)
- ✓ Model cache directory prepared (`./models`)
- ✓ Project structure verified

### How to Run the Project

#### Option 1: Local Python Server (Current Setup)
```bash
# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Start the FastAPI server
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# In another terminal, run the evaluation
python src/evaluation/run_evaluation.py
```

#### Option 2: Docker Compose (Recommended for Prod)
```bash
# Start all services
docker-compose up --build

# In another terminal, run evaluation
docker exec -it pidima-llm-api python src/evaluation/run_evaluation.py
```

### API Endpoints (Once Server Starts)

1. **Health Check** - `GET /health`
   ```bash
   curl http://localhost:8000/health
   ```

2. **Generate Text** - `POST /generate`
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is AI?", "max_tokens": 100}'
   ```

3. **Evaluate MCQ** - `POST /evaluate_mcq`
   ```bash
   curl -X POST http://localhost:8000/evaluate_mcq \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the capital of France?",
       "options": ["London", "Berlin", "Paris", "Madrid"]
     }'
   ```

4. **Interactive API Docs** - Visit `http://localhost:8000/docs`

### Project Architecture

- **Model**: microsoft/Phi-3-mini-4k-instruct (3.8B parameters)
- **Framework**: FastAPI + Transformers + PyTorch
- **Dataset**: MMLU (Massive Multitask Language Understanding)
- **Evaluation**: 150 questions across 10 categories

### Running Evaluation

Once the API server is running, in a separate terminal:
```bash
python src/evaluation/run_evaluation.py
```

This will:
1. Download MMLU dataset from HuggingFace
2. Evaluate 150 questions through the API
3. Generate comprehensive metrics and visualizations
4. Create error analysis report in `./results/`

### Expected Performance

- Overall Accuracy: 65-75%
- Response Time: 200-300ms per question
- Total Evaluation Time: ~10-20 minutes
- Memory Usage: 4-6 GB
- Docker Image Size: 2-3 GB

### Project Files

Key source files:
- `src/api/main.py` - FastAPI application with 3 endpoints
- `src/llm/loader.py` - Model loading and optimization
- `src/llm/inference.py` - Inference engine for text generation
- `src/evaluation/run_evaluation.py` - Evaluation pipeline
- `src/evaluation/metrics.py` - Metrics calculation

### Configuration

The project uses environment variables for configuration. Create a `.env` file from the template:
```bash
cp env.sh .env
```

Key settings:
- `MODEL_NAME`: HuggingFace model identifier
- `DEVICE`: cpu or cuda
- `EVAL_BATCH_SIZE`: Number of concurrent requests (default: 5)
- `LOG_LEVEL`: INFO, DEBUG, etc.

### Troubleshooting

**If model loading is slow:**
- This is normal for CPU inference of a 3.8B model
- First run may take 5-10 minutes to download and initialize the model
- Subsequent runs will be faster due to caching

**Memory issues:**
- Ensure 8GB+ RAM available
- Reduce `EVAL_BATCH_SIZE` in .env if needed
- Consider using GPU with CUDA for faster inference

**API not responding:**
- Wait longer for model to load (check logs)
- Increase timeout values in configuration
- Check firewall/port availability

### Next Steps

1. Start the API server using one of the options above
2. Wait for "Model loaded successfully!" message
3. Test health endpoint: `curl http://localhost:8000/health`
4. Run the evaluation pipeline
5. Check results in `./results/` directory

---

**Project**: Pidima AI Engineer Challenge - LLM Evaluation Pipeline
**Candidate**: Mohamed Gamal Elbayoumi
**Setup Date**: November 14, 2025
