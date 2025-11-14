# API Documentation

Complete API reference for the Pidima LLM Evaluation API.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. In production, implement API keys or OAuth2.

---

## Endpoints

### 1. Root Endpoint

Get API information and status.

**Endpoint**: `GET /`

**Response**:
```json
{
  "name": "Pidima LLM Evaluation API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

---

### 2. Health Check

Check API and model status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "microsoft/Phi-3-mini-4k-instruct",
  "uptime_seconds": 3600
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### 3. Generate Text

Generate text completion from a prompt.

**Endpoint**: `POST /generate`

**Request Body**:
```json
{
  "prompt": "string (required, 1-2048 chars)",
  "max_tokens": "integer (optional, default: 100, range: 1-2048)",
  "temperature": "float (optional, default: 0.7, range: 0.0-2.0)",
  "top_p": "float (optional, default: 0.9, range: 0.0-1.0)",
  "top_k": "integer (optional, default: 50, range: 0-100)"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Response**:
```json
{
  "text": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, enabling parallel processing of information...",
  "tokens": 42,
  "time_ms": 234
}
```

**Status Codes**:
- `200 OK`: Generation successful
- `400 Bad Request`: Invalid parameters
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Generation failed

---

### 4. Evaluate MCQ

Evaluate a multiple-choice question and return the predicted answer.

**Endpoint**: `POST /evaluate_mcq`

**Request Body**:
```json
{
  "question": "string (required, 5-2048 chars)",
  "options": "array of strings (required, 2-26 items)",
  "context": "string (optional, max 4096 chars)"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/evaluate_mcq \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the powerhouse of the cell?",
    "options": [
      "Nucleus",
      "Mitochondria",
      "Ribosome",
      "Golgi apparatus"
    ],
    "context": null
  }'
```

**Response**:
```json
{
  "answer": "B",
  "confidence": 0.87,
  "reasoning": "Mitochondria are known as the powerhouse of the cell",
  "time_ms": 189
}
```

**With Context Example**:
```bash
curl -X POST http://localhost:8000/evaluate_mcq \
  -H "Content-Type: application/json" \
  -d '{
    "question": "According to the passage, what was the main cause?",
    "options": [
      "Economic factors",
      "Political instability",
      "Natural disaster",
      "Military conflict"
    ],
    "context": "The passage describes how political instability led to widespread changes..."
  }'
```

**Status Codes**:
- `200 OK`: Evaluation successful
- `400 Bad Request`: Invalid options count or parameters
- `422 Unprocessable Entity`: Validation error
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Evaluation failed

---

## Interactive Documentation

The API provides auto-generated interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- View request/response schemas
- Test endpoints directly in the browser
- Download OpenAPI specification

---

## Error Handling

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**400 Bad Request**:
```json
{
  "detail": "Options must contain 2-26 items"
}
```

**422 Validation Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

**503 Service Unavailable**:
```json
{
  "detail": "Model not loaded"
}
```

---

## Rate Limiting

Currently no rate limiting implemented. For production:
- Implement rate limiting (e.g., 100 requests/minute per IP)
- Add API key authentication
- Track usage metrics

---

## Performance Considerations

### Response Times

Typical response times (CPU inference):
- `/health`: < 10ms
- `/generate` (100 tokens): 200-500ms
- `/evaluate_mcq`: 150-300ms

GPU inference reduces times by 5-10x.

### Optimization Tips

1. **Batch Requests**: For multiple MCQs, send them concurrently (up to 5 at a time)
2. **Temperature**: Use `temperature=0.0` for deterministic results
3. **Token Limit**: Lower `max_tokens` for faster responses
4. **Model Caching**: First request is slower due to model loading

---

## Python Client Example

```python
import requests

class LLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()
    
    def evaluate_mcq(self, question, options, context=None):
        response = requests.post(
            f"{self.base_url}/evaluate_mcq",
            json={
                "question": question,
                "options": options,
                "context": context
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = LLMClient()

# Check health
print(client.health_check())

# Generate text
result = client.generate("What is AI?")
print(result["text"])

# Evaluate MCQ
answer = client.evaluate_mcq(
    question="What is 2+2?",
    options=["2", "3", "4", "5"]
)
print(f"Answer: {answer['answer']}")
```

---

## JavaScript/TypeScript Client Example

```typescript
class LLMClient {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  async generate(
    prompt: string,
    maxTokens: number = 100,
    temperature: number = 0.7
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: maxTokens,
        temperature
      })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async evaluateMCQ(
    question: string,
    options: string[],
    context?: string
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/evaluate_mcq`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, options, context })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
}

// Usage
const client = new LLMClient();

// Evaluate MCQ
const result = await client.evaluateMCQ(
  "What is the capital of France?",
  ["London", "Paris", "Berlin", "Madrid"]
);
console.log(`Answer: ${result.answer}`);
```

---

## Testing

Use the provided test suite:

```bash
pytest tests/test_api.py -v
```

Or test manually with curl:

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
  -d '{
    "question": "Test question?",
    "options": ["A", "B", "C", "D"]
  }'
```

---

## Security Considerations

For production deployment:

1. **HTTPS**: Always use TLS/SSL
2. **Authentication**: Implement API key or OAuth2
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Already implemented with Pydantic
5. **CORS**: Configure allowed origins (currently allows all)
6. **Logging**: Monitor for suspicious activity
7. **Resource Limits**: Set max request size and timeout

---

## Monitoring

Recommended metrics to track:

- Request rate (requests/second)
- Response times (p50, p95, p99)
- Error rates by endpoint
- Model inference time
- Memory usage
- CPU/GPU utilization

Consider integrating:
- Prometheus for metrics
- Grafana for dashboards
- ELK stack for log aggregation

---

## Changelog

### Version 1.0.0 (2025-11-08)
- Initial release
- Three core endpoints: /generate, /evaluate_mcq, /health
- Phi-3-mini model support
- Docker deployment
- Comprehensive evaluation pipeline