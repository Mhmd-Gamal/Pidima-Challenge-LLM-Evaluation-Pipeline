#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for loading the Phi-3 model.
"""
import sys
import torch
import os
os.chdir(r"c:\Users\Target\Documents\Pidima")

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nLoading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir="./models",
    trust_remote_code=True
)
print("[OK] Tokenizer loaded successfully")

print("\nLoading model (this may take a while on CPU)...")
from transformers import AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        cache_dir="./models",
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map="cpu"
    )
    print("[OK] Model loaded successfully")
    print("[OK] Model moved to CPU")
    model.eval()
    print("[OK] Model set to evaluation mode")
    
    # Simple test inference
    print("\nTesting simple inference...")
    inputs = tokenizer("What is 2+2?", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[OK] Inference successful: {result[:50]}...")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")

