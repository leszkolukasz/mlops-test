# Contributing Guide

Thank you for your interest in improving this MLOps pipeline project! This guide will help you contribute effectively.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlops-test.git
cd mlops-test

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Verify setup
python validate_setup.py
```

## Code Style

### Python
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

**Example:**
```python
def process_data(data: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Process and split data for training.
    
    Args:
        data: Input DataFrame with features and target
        test_size: Proportion of data for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Implementation
    pass
```

### Documentation
- Use Markdown for documentation
- Include code examples
- Keep language clear and concise
- Add diagrams where helpful

## Project Structure

When adding new features, follow the existing structure:

```
src/
├── features/          # Feature engineering code
├── models/            # Model training code
└── utils/             # Utility functions (if needed)

deployments/
├── ray_serve/         # Ray Serve deployment
├── bentoml/           # BentoML deployment
└── <new_framework>/   # Add new deployment options here

docs/
└── <new_topic>.md     # Add new documentation here
```

## Adding New Features

### 1. New Feature Engineering

Add to `src/features/feature_engineering.py`:

```python
def create_new_feature(self, df):
    """Create your new feature"""
    df['new_feature'] = df['col1'] * df['col2']
    return df
```

Update the `create_features` method to call your new function.

### 2. New Model Type

Create a new file in `src/models/`:

```python
# src/models/my_new_model.py
class MyNewModel:
    def __init__(self):
        # Initialize
        pass
    
    def train(self, X_train, y_train):
        # Train model
        pass
    
    def predict(self, X):
        # Make predictions
        pass
```

Update `pipeline.py` to use your new model.

### 3. New Deployment Option

Create a new directory in `deployments/`:

```bash
mkdir deployments/my_framework
cd deployments/my_framework
```

Add required files:
- `deploy.py` - Deployment script
- `deploy.sh` - Shell script for production
- `Dockerfile` - Container definition (optional)
- `README.md` - Framework-specific documentation

Update `.github/workflows/mlops-pipeline.yml` to add deployment job.

### 4. New Documentation

Create a new Markdown file in `docs/`:

```bash
touch docs/my_new_topic.md
```

Link to it from the main README.md.

## Testing

### Unit Tests

If adding unit tests, create them in a `tests/` directory:

```bash
mkdir -p tests
touch tests/test_features.py
```

Example test:
```python
import unittest
from src.features.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def test_feature_creation(self):
        engineer = FeatureEngineer()
        df = engineer.load_data()
        df = engineer.create_features(df)
        
        # Check new features exist
        self.assertIn('petal_area', df.columns)
        self.assertIn('sepal_area', df.columns)
```

### Integration Tests

Test the entire pipeline:

```bash
# Run pipeline with minimal trials
python pipeline.py run --n-trials 2

# Validate output
test -f models/iris_model.pkl || echo "Model not created"
test -f models/scaler.pkl || echo "Scaler not created"
```

### Deployment Tests

```bash
# Test deployment script
./test_deployment.sh
```

## Pull Request Process

1. **Update Documentation**: Ensure all documentation is updated
2. **Add Tests**: Include tests for new features
3. **Run Validation**: Run `python validate_setup.py`
4. **Update Changelog**: Add entry to CHANGELOG.md (if it exists)
5. **Submit PR**: Create pull request with clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Ran validation script
- [ ] Tested locally
- [ ] Updated tests (if applicable)
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Common Contributions

### Easy (Good First Issues)
- Fix typos in documentation
- Add more examples
- Improve error messages
- Add input validation
- Update dependencies

### Medium
- Add new feature engineering techniques
- Implement different ML models
- Improve logging
- Add data validation
- Create visualization utilities

### Advanced
- Add new deployment framework
- Implement A/B testing
- Add model monitoring
- Implement feature store
- Add Kubernetes support
- Integrate drift detection

## Specific Contribution Ideas

### Feature Engineering
- Add more feature interactions
- Implement feature selection
- Add dimensionality reduction
- Time-based features (if applicable)

### Model Training
- Support for other algorithms (XGBoost, LightGBM)
- Ensemble methods
- Neural networks
- Transfer learning

### Deployment
- Add TorchServe deployment
- Add TensorFlow Serving
- Add KServe/Seldon
- Add Triton Inference Server

### Monitoring
- Add Prometheus metrics
- Implement Grafana dashboards
- Add model drift detection
- Performance monitoring

### Data
- Add data validation (Great Expectations)
- Implement data versioning (DVC)
- Add feature store (Feast)
- Data quality checks

### Infrastructure
- Add Kubernetes configurations
- Terraform/CloudFormation templates
- Add Helm charts
- CI/CD improvements

## Code Review Process

All submissions require review. We look for:

1. **Functionality**: Does it work as intended?
2. **Code Quality**: Is it readable and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is it well documented?
5. **Style**: Does it follow project conventions?

## Communication

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **PR Comments**: Use for code review feedback

## Recognition

Contributors will be:
- Listed in the contributors section
- Credited in release notes
- Mentioned in relevant documentation

## Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search closed issues/PRs
3. Open a new issue with the "question" label

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Respect different viewpoints

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to this MLOps pipeline project! Your contributions help make this a better learning resource for everyone.
