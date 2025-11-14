"""
Main evaluation pipeline for MCQ benchmarking.
"""
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import aiohttp

from .dataset import MCQDataset
from .metrics import MetricsCalculator
from ..utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates the complete MCQ evaluation process."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        results_dir: str = "./results"
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            api_url: Base URL for the LLM API
            results_dir: Directory to save results
        """
        self.api_url = api_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        
    async def check_api_health(self) -> bool:
        """
        Check if the API is healthy and ready.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"API Health: {data}")
                        return data.get("model_loaded", False)
                    return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def evaluate_question(
        self,
        session: aiohttp.ClientSession,
        question_data: Dict[str, Any],
        question_id: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through the API.
        
        Args:
            session: aiohttp session
            question_data: Question dictionary
            question_id: Unique question identifier
            
        Returns:
            Result dictionary with prediction and metadata
        """
        try:
            # Prepare request
            request_data = {
                "question": question_data["question"],
                "options": question_data["options"],
                "context": question_data.get("context")
            }
            
            # Call API
            async with session.post(
                f"{self.api_url}/evaluate_mcq",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Add metadata
                    result["question_id"] = question_id
                    result["question"] = question_data["question"]
                    result["options"] = question_data["options"]
                    result["correct_answer"] = question_data["correct_answer"]
                    result["category"] = question_data.get("category", "unknown")
                    result["predicted_answer"] = result.get("answer", "")
                    result["is_correct"] = (
                        result["predicted_answer"] == question_data["correct_answer"]
                    )
                    
                    return result
                else:
                    logger.error(f"API returned status {response.status} for question {question_id}")
                    return self._create_error_result(question_data, question_id, "API error")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout for question {question_id}")
            return self._create_error_result(question_data, question_id, "Timeout")
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {e}")
            return self._create_error_result(question_data, question_id, str(e))
    
    def _create_error_result(
        self,
        question_data: Dict[str, Any],
        question_id: int,
        error: str
    ) -> Dict[str, Any]:
        """Create a result dictionary for failed evaluations."""
        return {
            "question_id": question_id,
            "question": question_data["question"],
            "options": question_data["options"],
            "correct_answer": question_data["correct_answer"],
            "category": question_data.get("category", "unknown"),
            "predicted_answer": "",
            "is_correct": False,
            "confidence": 0.0,
            "reasoning": "",
            "time_ms": 0,
            "error": error
        }
    
    async def run_evaluation(
        self,
        n_samples: int = 150,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            n_samples: Number of questions to evaluate
            batch_size: Number of concurrent API requests
            
        Returns:
            List of evaluation results
        """
        logger.info("Starting evaluation pipeline")
        
        # Check API health
        logger.info("Checking API health...")
        is_healthy = await self.check_api_health()
        if not is_healthy:
            raise RuntimeError("API is not healthy. Please start the API server first.")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = MCQDataset()
        questions = dataset.sample_questions(n_samples=n_samples)
        logger.info(f"Loaded {len(questions)} questions")
        
        # Evaluate questions
        logger.info("Evaluating questions...")
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Process in batches for rate limiting
            for i in tqdm(range(0, len(questions), batch_size), desc="Evaluating"):
                batch = questions[i:i+batch_size]
                
                # Create tasks for concurrent evaluation
                tasks = [
                    self.evaluate_question(session, q, i+j)
                    for j, q in enumerate(batch)
                ]
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                # Save intermediate results every 25 questions
                if (i + batch_size) % 25 == 0:
                    self._save_results(results, suffix="_intermediate")
        
        # Save final results
        logger.info("Saving results...")
        self._save_results(results)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        calculator = MetricsCalculator(results)
        metrics = calculator.calculate_all_metrics()
        
        # Save metrics
        metrics_path = self.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, indent=2, fp=f)
        
        logger.info(f"Evaluation complete. Results saved to {self.results_dir}")
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], suffix: str = ""):
        """Save evaluation results to JSON file."""
        filename = f"evaluation_results{suffix}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        logger.info(f"Saved results to {filepath}")


async def main():
    """Main entry point for evaluation script."""
    pipeline = EvaluationPipeline()
    
    try:
        results = await pipeline.run_evaluation(n_samples=150)
        
        # Generate error analysis report
        logger.info("Generating error analysis report...")
        calculator = MetricsCalculator(results)
        calculator.generate_visualizations(str(pipeline.results_dir / "visualizations"))
        calculator.generate_error_report(str(pipeline.results_dir / "error_analysis.md"))
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())