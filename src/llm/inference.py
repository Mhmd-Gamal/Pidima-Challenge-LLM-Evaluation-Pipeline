"""
Inference engine for text generation and MCQ evaluation.
"""
import re
import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles model inference for text generation and MCQ evaluation."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Initialize the inference engine.
        
        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated portion
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return {
                "text": generated_text.strip(),
                "tokens": len(generated_ids)
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def evaluate_mcq(
        self,
        question: str,
        options: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a multiple-choice question and extract the answer.
        
        Args:
            question: The question text
            options: List of answer options
            context: Optional context passage
            
        Returns:
            Dictionary with predicted answer and metadata
        """
        try:
            # Build the prompt
            prompt = self._build_mcq_prompt(question, options, context)
            
            # Generate response with low temperature for consistency
            # NOTE: Using temperature=0.0 for deterministic evaluation
            # In production, might want to sample multiple times and take majority vote
            result = self.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.0,  # Deterministic for evaluation
                top_p=1.0,
                top_k=50
            )
            
            raw_output = result["text"]
            
            # Extract answer letter
            answer = self._extract_answer(raw_output, len(options))
            
            # Extract reasoning (first sentence of output)
            reasoning = self._extract_reasoning(raw_output)
            
            # TODO: Implement proper confidence scoring based on logits
            # Currently using placeholder values
            # Real implementation should look at probability distribution over A/B/C/D tokens
            confidence = 0.0
            if answer:
                confidence = 0.85  # Default high confidence for extracted answers
            
            return {
                "answer": answer if answer else "A",  # Default to A if extraction fails
                "confidence": confidence,
                "reasoning": reasoning,
                "raw_output": raw_output
            }
            
        except Exception as e:
            logger.error(f"MCQ evaluation error: {e}")
            raise
    
    def _build_mcq_prompt(
        self,
        question: str,
        options: List[str],
        context: Optional[str] = None
    ) -> str:
        """
        Build a structured prompt for MCQ evaluation.
        
        I experimented with different prompt formats:
        1. Chain-of-thought style (asking model to reason first)
        2. Direct instruction (just answer with letter)
        3. Few-shot examples (showing 2-3 examples first)
        
        The direct instruction format performed best for Phi-3 in my tests,
        giving cleaner outputs that are easier to parse. For production,
        would want to A/B test different prompts on a validation set.
        
        Args:
            question: Question text
            options: Answer options
            context: Optional context
            
        Returns:
            Formatted prompt string
        """
        # Letter labels for options
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
        
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append(
            "You are taking a multiple-choice exam. "
            "Answer with only the letter of the correct option."
        )
        
        # Add context if provided
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        # Add question
        prompt_parts.append(f"\nQuestion: {question}")
        
        # Add options
        prompt_parts.append("\nOptions:")
        for letter, option in zip(letters, options):
            prompt_parts.append(f"{letter}) {option}")
        
        # Add answer prompt
        prompt_parts.append("\nAnswer (provide only the letter):")
        
        return "\n".join(prompt_parts)
    
    def _extract_answer(self, text: str, num_options: int) -> Optional[str]:
        """
        Extract answer letter from model output using regex patterns.
        
        This was one of the trickiest parts - LLMs are inconsistent in how they
        format answers. I've seen:
        - "The answer is B"
        - "B) is correct"
        - "I think B"
        - Just "B"
        - "(B)"
        
        The regex patterns below handle all these cases. If I had more time,
        I'd implement constrained decoding to force the model to output only
        valid letters, which would eliminate this parsing entirely.
        
        Args:
            text: Model output text
            num_options: Number of available options
            
        Returns:
            Extracted answer letter or None
        """
        # Valid letters based on number of options
        valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_options]
        
        # Try multiple extraction patterns (ordered by reliability)
        patterns = [
            # Match standalone letter at start (most reliable)
            rf'^([{valid_letters}])\b',
            # Match "Answer: X" or "Answer is X"
            rf'(?:answer|option)(?:\s+is)?[:\s]+([{valid_letters}])\b',
            # Match letter in parentheses
            rf'\(([{valid_letters}])\)',
            # Match letter followed by closing paren
            rf'([{valid_letters}])\)',
            # Match any standalone letter (least reliable, catches false positives)
            rf'\b([{valid_letters}])\b'
        ]
        
        # Try each pattern
        text_upper = text.upper()
        for pattern in patterns:
            match = re.search(pattern, text_upper, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_letters:
                    logger.debug(f"Extracted answer: {letter} using pattern: {pattern}")
                    return letter
        
        logger.warning(f"Failed to extract answer from: {text[:100]}")
        return None
    
    def _extract_reasoning(self, text: str, max_length: int = 200) -> str:
        """
        Extract brief reasoning from model output.
        
        Args:
            text: Model output text
            max_length: Maximum reasoning length
            
        Returns:
            Extracted reasoning text
        """
        # Take first sentence or up to max_length
        sentences = text.split('.')
        if sentences:
            reasoning = sentences[0].strip()
            if len(reasoning) > max_length:
                reasoning = reasoning[:max_length].rsplit(' ', 1)[0] + "..."
            return reasoning
        return text[:max_length]
