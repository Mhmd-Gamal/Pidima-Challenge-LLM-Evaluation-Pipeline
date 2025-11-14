"""
Dataset loading and preprocessing for MCQ evaluation.
"""
import logging
import random
from typing import List, Dict, Any
from datasets import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class MCQDataset:
    """Handles loading and sampling of MCQ datasets."""
    
    def __init__(self, dataset_name: str = "cais/mmlu", subset: str = "all"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            subset: Specific subset to load (or 'all' for sampling across subsets)
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None
        
    def load(self):
        """Load the dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            if self.dataset_name == "cais/mmlu":
                # MMLU has multiple subsets
                self.dataset = load_dataset(self.dataset_name, "all")
            else:
                self.dataset = load_dataset(self.dataset_name)
                
            logger.info("Dataset loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def sample_questions(
        self,
        n_samples: int = 150,
        categories: List[str] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Sample questions from the dataset with stratified sampling.
        
        Args:
            n_samples: Total number of questions to sample
            categories: Specific categories to sample from (None for all)
            seed: Random seed for reproducibility
            
        Returns:
            List of question dictionaries
        """
        random.seed(seed)
        
        if self.dataset is None:
            self.load()
        
        questions = []
        
        try:
            if self.dataset_name == "cais/mmlu":
                questions = self._sample_mmlu(n_samples, categories, seed)
            else:
                questions = self._sample_generic(n_samples, seed)
                
            logger.info(f"Sampled {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to sample questions: {e}")
            raise
    
    def _sample_mmlu(
        self,
        n_samples: int,
        categories: List[str] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Sample from MMLU dataset with category balancing."""
        
        # Default categories for balanced sampling
        if categories is None:
            categories = [
                "abstract_algebra",
                "anatomy",
                "astronomy",
                "business_ethics",
                "clinical_knowledge",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "econometrics",
                "electrical_engineering",
                "elementary_mathematics",
                "formal_logic",
                "global_facts",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_european_history",
                "high_school_geography",
                "high_school_government_and_politics",
                "high_school_macroeconomics",
                "high_school_mathematics",
                "high_school_microeconomics",
                "high_school_physics",
                "high_school_psychology",
                "high_school_statistics",
                "high_school_us_history",
                "high_school_world_history",
                "human_aging",
                "human_sexuality",
                "international_law",
                "jurisprudence",
                "logical_fallacies",
                "machine_learning",
                "management",
                "marketing",
                "medical_genetics",
                "miscellaneous",
                "moral_disputes",
                "moral_scenarios",
                "nutrition",
                "philosophy",
                "prehistory",
                "professional_accounting",
                "professional_law",
                "professional_medicine",
                "professional_psychology",
                "public_relations",
                "security_studies",
                "sociology",
                "us_foreign_policy",
                "virology",
                "world_religions"
            ]
        
        # Sample evenly from selected categories
        samples_per_category = max(1, n_samples // len(categories[:10]))  # Use top 10 categories
        selected_categories = categories[:10]
        
        questions = []
        
        for category in selected_categories:
            try:
                # Load test split for this category
                subset_data = load_dataset(
                    self.dataset_name,
                    category,
                    split="test"
                )
                
                # Sample from this category
                n_from_category = min(samples_per_category, len(subset_data))
                indices = random.sample(range(len(subset_data)), n_from_category)
                
                for idx in indices:
                    item = subset_data[idx]
                    
                    # Format question
                    question_dict = {
                        "question": item["question"],
                        "options": item["choices"],
                        "correct_answer": chr(65 + item["answer"]),  # Convert 0,1,2,3 to A,B,C,D
                        "category": category,
                        "context": None
                    }
                    questions.append(question_dict)
                    
                    if len(questions) >= n_samples:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to load category {category}: {e}")
                continue
            
            if len(questions) >= n_samples:
                break
        
        # Shuffle questions
        random.shuffle(questions)
        
        return questions[:n_samples]
    
    def _sample_generic(self, n_samples: int, seed: int) -> List[Dict[str, Any]]:
        """Sample from generic MCQ dataset."""
        
        questions = []
        
        # Get test split
        if "test" in self.dataset:
            data = self.dataset["test"]
        elif "validation" in self.dataset:
            data = self.dataset["validation"]
        else:
            data = self.dataset["train"]
        
        # Sample indices
        n_from_dataset = min(n_samples, len(data))
        indices = random.sample(range(len(data)), n_from_dataset)
        
        for idx in indices:
            item = data[idx]
            
            # Try to format question (adjust based on dataset structure)
            try:
                question_dict = {
                    "question": item.get("question", item.get("text", "")),
                    "options": item.get("choices", item.get("options", [])),
                    "correct_answer": item.get("answer", item.get("label", "A")),
                    "category": item.get("category", "general"),
                    "context": item.get("context", None)
                }
                questions.append(question_dict)
                
            except Exception as e:
                logger.warning(f"Failed to parse question at index {idx}: {e}")
                continue
        
        return questions