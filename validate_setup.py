#!/usr/bin/env python
"""
Validate the MLOps project setup.
Checks that all required files exist and have correct structure.
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} - NOT FOUND")
        return False

def validate_project():
    """Validate the project structure"""
    print("MLOps Project Validation")
    print("=" * 50)
    
    checks_passed = 0
    checks_total = 0
    
    # Check main files
    main_files = [
        ("pipeline.py", "Metaflow pipeline"),
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Main documentation"),
        ("setup.sh", "Setup script"),
        (".gitignore", "Git ignore file"),
    ]
    
    print("\nMain Files:")
    for filepath, description in main_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check directories
    directories = [
        ("src", "Source code directory"),
        ("src/features", "Features directory"),
        ("src/models", "Models directory"),
        ("deployments", "Deployments directory"),
        ("deployments/ray_serve", "Ray Serve deployment"),
        ("deployments/bentoml", "BentoML deployment"),
        ("config", "Configuration directory"),
        ("docs", "Documentation directory"),
        (".github/workflows", "GitHub Actions workflows"),
    ]
    
    print("\nDirectories:")
    for dirpath, description in directories:
        checks_total += 1
        if check_directory_exists(dirpath, description):
            checks_passed += 1
    
    # Check source files
    source_files = [
        ("src/__init__.py", "Source package init"),
        ("src/features/__init__.py", "Features package init"),
        ("src/features/feature_engineering.py", "Feature engineering module"),
        ("src/models/__init__.py", "Models package init"),
        ("src/models/train_model.py", "Model training module"),
    ]
    
    print("\nSource Files:")
    for filepath, description in source_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check deployment files
    deployment_files = [
        ("deployments/ray_serve/deploy.py", "Ray Serve deployment script"),
        ("deployments/ray_serve/deploy.sh", "Ray Serve shell script"),
        ("deployments/ray_serve/Dockerfile", "Ray Serve Dockerfile"),
        ("deployments/bentoml/service.py", "BentoML service"),
        ("deployments/bentoml/save_model.py", "BentoML save script"),
        ("deployments/bentoml/deploy.sh", "BentoML shell script"),
        ("deployments/bentoml/bentofile.yaml", "BentoML config"),
    ]
    
    print("\nDeployment Files:")
    for filepath, description in deployment_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check documentation
    doc_files = [
        ("docs/architecture.md", "Architecture documentation"),
        ("docs/deployment_comparison.md", "Deployment comparison"),
        ("docs/quickstart.md", "Quick start guide"),
    ]
    
    print("\nDocumentation:")
    for filepath, description in doc_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check GitHub Actions
    workflow_files = [
        (".github/workflows/mlops-pipeline.yml", "MLOps pipeline workflow"),
    ]
    
    print("\nGitHub Actions:")
    for filepath, description in workflow_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check config
    config_files = [
        ("config/config.yaml", "Configuration file"),
    ]
    
    print("\nConfiguration:")
    for filepath, description in config_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Validation Results: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("✓ All checks passed! Project is properly set up.")
        return 0
    else:
        print(f"✗ {checks_total - checks_passed} checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(validate_project())
