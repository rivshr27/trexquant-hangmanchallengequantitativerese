#!/usr/bin/env python3
"""
Test script to verify the hangman project setup is working correctly.
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

def test_environment():
    """Test the Python environment and required packages."""
    # Test Python version
    python_version = sys.version_info
    
    # Test PyTorch and NumPy availability
    torch_available = True
    numpy_available = True
    
    try:
        torch.__version__
        torch.cuda.is_available()
    except:
        torch_available = False
    
    try:
        np.__version__
    except:
        numpy_available = False

def test_project_structure():
    """Test that all required directories and files exist."""
    required_dirs = ['dataset', 'deeplearning', 'models', 'plots', 'pickle']
    required_files = ['utils.py', 'hangman_local.ipynb', 'deeplearning/config.yaml']
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import utils
        
        from deeplearning.models import RNN
        
        from train import dl_model
        
        # Test n-gram functionality
        test_words = ['apple', 'banana', 'cherry']
        n_grams = utils.build_n_gram(test_words, 3)
        
    except ImportError as e:
        return False
    
    return True

def test_config():
    """Test the configuration file."""
    try:
        with open("deeplearning/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify essential config keys exist
        required_keys = ['dataset', 'models', 'cuda']
        for key in required_keys:
            if key not in config:
                return False
        
    except Exception as e:
        return False
    
    return True

def test_dataset():
    """Test that dataset files exist."""
    dataset_files = [
        'dataset/words_250000_train.txt',
        'dataset/words_not_contained.txt'
    ]
    
    missing_files = []
    for file_path in dataset_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Run all tests."""
    test_environment()
    structure_ok = test_project_structure()
    imports_ok = test_imports()
    config_ok = test_config()
    dataset_ok = test_dataset()
    
    all_tests_passed = structure_ok and imports_ok and config_ok and dataset_ok
    
    if all_tests_passed:
        return True
    else:
        return False

if __name__ == "__main__":
    main()