"""
Model loading utilities with caching and optimization.
"""
import logging
import torch
from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles efficient loading of LLM models with quantization support."""
    
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        use_4bit: bool = True
    ):
        """
        Initialize the model loader.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache model weights
            device: Device to load model on (cpu/cuda)
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.use_4bit = use_4bit and device == "cuda"
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer with appropriate optimizations.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"4-bit quantization: {self.use_4bit}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if using GPU
            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            
            # Load model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "quantization_config": quantization_config,
                "trust_remote_code": True,
                "dtype": torch.float32 if self.device == "cpu" else torch.float16,
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager"  # Avoid flash-attention compatibility issues
            }
            
            # Add device_map for both CPU and GPU
            model_kwargs["device_map"] = "cpu" if self.device == "cpu" else "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model parameters: {self.count_parameters()/1e6:.1f}M")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def count_parameters(self) -> int:
        """
        Count the total number of model parameters.
        
        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")