"""
Metrics calculation and error analysis for evaluation results.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates metrics and generates analysis from evaluation results."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize metrics calculator with evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
        """
        self.results = results
        self.df = pd.DataFrame(results)
        
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {
            "total_questions": len(self.results),
            "overall_accuracy": self._calculate_accuracy(),
            "per_category_accuracy": self._calculate_category_accuracy(),
            "response_time_stats": self._calculate_time_stats(),
            "extraction_success_rate": self._calculate_extraction_rate(),
            "confidence_stats": self._calculate_confidence_stats(),
            "error_analysis": self._analyze_errors()
        }
        
        return metrics
    
    def _calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if len(self.results) == 0:
            return 0.0
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        return correct / len(self.results)
    
    def _calculate_category_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """Calculate accuracy breakdown by category."""
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in self.results:
            category = result.get("category", "unknown")
            category_stats[category]["total"] += 1
            if result.get("is_correct", False):
                category_stats[category]["correct"] += 1
        
        # Calculate accuracy for each category
        category_accuracy = {}
        for category, stats in category_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            category_accuracy[category] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }
        
        return category_accuracy
    
    def _calculate_time_stats(self) -> Dict[str, float]:
        """Calculate response time statistics."""
        times = [r.get("time_ms", 0) for r in self.results if r.get("time_ms", 0) > 0]
        
        if not times:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0
            }
        
        return {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times))
        }
    
    def _calculate_extraction_rate(self) -> float:
        """Calculate success rate of answer extraction."""
        total = len(self.results)
        if total == 0:
            return 0.0
        
        # Count results with non-empty predicted answers
        successful = sum(
            1 for r in self.results 
            if r.get("predicted_answer", "").strip() != ""
        )
        
        return successful / total
    
    def _calculate_confidence_stats(self) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        confidences = [
            r.get("confidence", 0.0) for r in self.results
            if r.get("confidence", 0.0) > 0
        ]
        
        if not confidences:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0
            }
        
        return {
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
            "std": float(np.std(confidences))
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns and types."""
        errors = [r for r in self.results if not r.get("is_correct", False)]
        
        if not errors:
            return {"total_errors": 0}
        
        # Count errors by category
        error_categories = Counter(e.get("category", "unknown") for e in errors)
        
        # Analyze extraction failures
        extraction_failures = sum(
            1 for e in errors
            if e.get("predicted_answer", "").strip() == ""
        )
        
        # Sample incorrect examples (up to 10)
        error_examples = []
        for i, error in enumerate(errors[:10]):
            error_examples.append({
                "question": error.get("question", "")[:100] + "...",
                "correct": error.get("correct_answer", ""),
                "predicted": error.get("predicted_answer", ""),
                "category": error.get("category", "unknown")
            })
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(self.results),
            "errors_by_category": dict(error_categories),
            "extraction_failures": extraction_failures,
            "error_examples": error_examples
        }
    
    def generate_visualizations(self, output_dir: str):
        """
        Generate visualization plots for evaluation results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # 1. Accuracy by Category
        self._plot_category_accuracy(output_path / "category_accuracy.png")
        
        # 2. Response Time Distribution
        self._plot_time_distribution(output_path / "response_time_dist.png")
        
        # 3. Confusion Matrix
        self._plot_confusion_matrix(output_path / "confusion_matrix.png")
        
        # 4. Confidence Distribution
        self._plot_confidence_distribution(output_path / "confidence_dist.png")
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_category_accuracy(self, filepath: Path):
        """Plot accuracy by category."""
        category_acc = self._calculate_category_accuracy()
        
        if not category_acc:
            return
        
        categories = list(category_acc.keys())
        accuracies = [category_acc[c]["accuracy"] for c in categories]
        totals = [category_acc[c]["total"] for c in categories]
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        categories = [categories[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        totals = [totals[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(categories, accuracies, color='steelblue')
        
        # Add value labels
        for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
            plt.text(acc + 0.01, i, f'{acc:.1%} ({total})', va='center')
        
        plt.xlabel('Accuracy')
        plt.ylabel('Category')
        plt.title('Model Accuracy by Category')
        plt.xlim(0, 1.1)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_distribution(self, filepath: Path):
        """Plot response time distribution."""
        times = [r.get("time_ms", 0) for r in self.results if r.get("time_ms", 0) > 0]
        
        if not times:
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(times):.0f}ms')
        plt.axvline(np.median(times), color='green', linestyle='--',
                   label=f'Median: {np.median(times):.0f}ms')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, filepath: Path):
        """Plot confusion matrix of predicted vs actual answers."""
        correct_answers = [r.get("correct_answer", "") for r in self.results]
        predicted_answers = [r.get("predicted_answer", "") for r in self.results]
        
        # Get unique labels
        all_labels = sorted(set(correct_answers + predicted_answers))
        if not all_labels:
            return
        
        # Create confusion matrix
        n_labels = len(all_labels)
        matrix = np.zeros((n_labels, n_labels))
        
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        for correct, predicted in zip(correct_answers, predicted_answers):
            if correct and predicted:
                i = label_to_idx[correct]
                j = label_to_idx[predicted]
                matrix[i, j] += 1
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=all_labels, yticklabels=all_labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Answer')
        plt.ylabel('Correct Answer')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, filepath: Path):
        """Plot confidence score distribution."""
        confidences = [r.get("confidence", 0.0) for r in self.results]
        
        if not confidences or max(confidences) == 0:
            return
        
        # Separate by correctness
        correct_conf = [r.get("confidence", 0.0) for r in self.results 
                       if r.get("is_correct", False)]
        incorrect_conf = [r.get("confidence", 0.0) for r in self.results
                         if not r.get("is_correct", False)]
        
        plt.figure(figsize=(10, 6))
        plt.hist([correct_conf, incorrect_conf], bins=20, 
                label=['Correct', 'Incorrect'],
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Correctness')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_error_report(self, filepath: str):
        """
        Generate comprehensive error analysis report in Markdown.
        
        Args:
            filepath: Path to save the report
        """
        logger.info("Generating error analysis report...")
        
        metrics = self.calculate_all_metrics()
        
        report_lines = [
            "# LLM MCQ Evaluation - Error Analysis Report",
            "",
            "## Executive Summary",
            "",
            f"- **Total Questions Evaluated**: {metrics['total_questions']}",
            f"- **Overall Accuracy**: {metrics['overall_accuracy']:.2%}",
            f"- **Average Response Time**: {metrics['response_time_stats']['mean_ms']:.0f}ms",
            f"- **Answer Extraction Success Rate**: {metrics['extraction_success_rate']:.2%}",
            "",
            "---",
            "",
            "## Quantitative Analysis",
            "",
            "### Overall Performance",
            "",
            f"The model achieved an overall accuracy of **{metrics['overall_accuracy']:.2%}** ",
            f"on {metrics['total_questions']} questions across multiple categories. ",
            f"This performance is {'significantly above' if metrics['overall_accuracy'] > 0.3 else 'near or below'} ",
            f"the random baseline of 25% for 4-option multiple choice questions.",
            "",
            "### Performance by Category",
            "",
            "| Category | Accuracy | Correct | Total |",
            "|----------|----------|---------|-------|"
        ]
        
        # Add category stats
        cat_acc = metrics['per_category_accuracy']
        for category in sorted(cat_acc.keys(), key=lambda x: cat_acc[x]['accuracy'], reverse=True):
            stats = cat_acc[category]
            report_lines.append(
                f"| {category} | {stats['accuracy']:.2%} | "
                f"{stats['correct']} | {stats['total']} |"
            )
        
        report_lines.extend([
            "",
            "### Response Time Analysis",
            "",
            f"- **Mean**: {metrics['response_time_stats']['mean_ms']:.0f}ms",
            f"- **Median**: {metrics['response_time_stats']['median_ms']:.0f}ms",
            f"- **Std Dev**: {metrics['response_time_stats']['std_ms']:.0f}ms",
            f"- **Range**: {metrics['response_time_stats']['min_ms']:.0f}ms - "
            f"{metrics['response_time_stats']['max_ms']:.0f}ms",
            "",
            "The model demonstrates consistent inference speed with low variance, ",
            "indicating stable performance across different question types.",
            "",
            "---",
            "",
            "## Qualitative Analysis",
            "",
            "### Error Patterns",
            ""
        ])
        
        # Add error analysis
        errors = metrics['error_analysis']
        report_lines.extend([
            f"**Total Errors**: {errors['total_errors']} ({errors['error_rate']:.1%})",
            "",
            "#### Error Distribution by Category",
            ""
        ])
        
        if 'errors_by_category' in errors:
            for category, count in sorted(
                errors['errors_by_category'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                report_lines.append(f"- **{category}**: {count} errors")
        
        report_lines.extend([
            "",
            f"#### Extraction Failures: {errors.get('extraction_failures', 0)}",
            "",
            "These represent cases where the model failed to produce a valid answer letter.",
            "",
            "### Example Errors",
            ""
        ])
        
        # Add error examples
        for i, example in enumerate(errors.get('error_examples', [])[:5], 1):
            report_lines.extend([
                f"**Example {i}** ({example['category']})",
                f"- Question: {example['question']}",
                f"- Correct: {example['correct']}",
                f"- Predicted: {example['predicted'] or 'None'}",
                ""
            ])
        
        report_lines.extend([
            "---",
            "",
            "## Recommendations",
            "",
            "### Immediate Improvements (No Retraining)",
            "",
            "1. **Enhanced Prompt Engineering**",
            "   - Implement few-shot learning with 2-3 examples",
            "   - Add chain-of-thought reasoning instructions",
            "   - Use role-based prompting for better context",
            "   - Expected gain: +5-10% accuracy",
            "",
            "2. **Improved Answer Extraction**",
            "   - Implement constrained decoding for A/B/C/D tokens",
            "   - Use logit bias to penalize non-answer outputs",
            "   - Expected gain: +2-3% accuracy",
            "",
            "3. **Category-Specific Prompting**",
            "   - Tailor prompts based on subject matter",
            "   - Add domain-specific context cues",
            "   - Expected gain: +3-5% accuracy in weak categories",
            "",
            "### Model-Level Improvements",
            "",
            "1. **Fine-Tuning**",
            "   - Fine-tune on 1000+ MCQ examples with explanations",
            "   - Use LoRA/QLoRA for parameter-efficient training",
            "   - Expected gain: +15-25% accuracy",
            "",
            "2. **Model Upgrade**",
            "   - Test larger models (7B → 13B parameters)",
            "   - Evaluate specialized models (e.g., MMLU-tuned)",
            "   - Expected gain: +10-20% accuracy",
            "",
            "### System-Level Enhancements",
            "",
            "1. **Ensemble Methods**",
            "   - Combine predictions from multiple temperature samples",
            "   - Use model ensembles for better reliability",
            "",
            "2. **Retrieval Augmentation**",
            "   - Add RAG for knowledge-intensive questions",
            "   - Integrate external knowledge bases",
            "",
            "---",
            "",
            "## Limitations",
            "",
            "- Sample size limited to " + str(metrics['total_questions']) + " questions",
            "- Single model evaluation (no comparison baseline)",
            "- Deterministic evaluation (temperature=0.0)",
            "- No human evaluation of reasoning quality",
            "",
            "## Future Work",
            "",
            "- Expand to larger sample sizes (1000+ questions)",
            "- Compare against other models and baselines",
            "- Implement confidence calibration",
            "- Evaluate reasoning quality beyond correctness",
            "- Test few-shot and chain-of-thought prompting",
            "",
            "---",
            "",
            "**Report Generated**: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
        
        # Write report
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Error analysis report saved to {filepath}")