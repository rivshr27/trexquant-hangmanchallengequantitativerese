#!/usr/bin/env python3
"""
Verification script to ensure your AI setup is ready for Trexquant API submission
"""

import os
import sys

def check_files():
    """Check if all required files exist"""
    required_files = [
        "words_250000_train.txt",
        "utils.py",
        "deeplearning/models.py",
        "deeplearning/train.py",
        "deeplearning/config.yaml",
        "hangman_api_user.ipynb"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_models():
    """Check if trained models exist"""
    model_dirs = [
        "models/GRU_4_1024_26/",
        "models/"
    ]
    
    model_found = False
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            best_models = [f for f in files if f.startswith('best_')]
            if best_models:
                model_found = True
                break
    
    return model_found

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import utils
    except ImportError as e:
        return False
    
    try:
        from deeplearning.train import dl_model
    except ImportError as e:
        return False
    
    try:
        from deeplearning.models import RNN
    except ImportError as e:
        return False
    
    try:
        import torch
    except ImportError as e:
        return False
    
    try:
        import numpy as np
    except ImportError as e:
        return False
    
    return True

def test_model_loading():
    """Test if models can be loaded"""
    try:
        import utils
        from deeplearning.train import dl_model
        
        # Test n-gram loading
        n_grams = utils.build_n_gram_from_file("words_250000_train.txt")
        
        # Test neural network loading
        dl_mod = dl_model('test_one')
        
        return True
        
    except Exception as e:
        return False

def main():
    """Run all verification checks"""
    checks_passed = 0
    total_checks = 4
    
    if check_files():
        checks_passed += 1
    
    if check_models():
        checks_passed += 1
    
    if test_imports():
        checks_passed += 1
    
    if test_model_loading():
        checks_passed += 1
    
    all_passed = checks_passed == total_checks
    return all_passed

if __name__ == "__main__":
    main()