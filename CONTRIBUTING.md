# Contributing to Enhanced Semantic Document Search Engine

Thank you for your interest in contributing to this project! This guide will help you get started.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/winthrop1/LLM_Search_Engine.git
   cd semantic_llm
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Install Tesseract OCR** (see [USAGE.md](USAGE.md) for instructions)

## Development Workflow

### Setting Up Your Environment

1. **Create a `.env` file** based on the template in [USAGE.md](USAGE.md)
2. **Add test documents** to the `./data/` folder
3. **Run the application** to ensure everything works:
   ```bash
   python main.py
   ```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the coding standards below
3. **Test your changes** using the provided test scripts
4. **Commit your changes** with clear, descriptive messages
5. **Push to your fork** and create a pull request

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and reasonably short
- Use type hints where appropriate

### Code Structure

```python
"""Module docstring describing the module's purpose."""

import standard_library_modules
import third_party_modules
from local_modules import local_imports


class ExampleClass:
    """Class docstring explaining the class purpose."""
    
    def __init__(self, param: str) -> None:
        """Initialize with clear parameter descriptions."""
        self.param = param
    
    def example_method(self, input_data: str) -> str:
        """
        Method docstring explaining what it does.
        
        Args:
            input_data: Description of the parameter
            
        Returns:
            Description of return value
        """
        return processed_data
```

### Documentation

- Update relevant documentation when adding features
- Include usage examples for new functionality
- Update the [ROADMAP.md](ROADMAP.md) if adding major features
- Keep [USAGE.md](USAGE.md) current with new capabilities

## Testing

### Running Tests

```bash
# Test basic functionality
python test_search_only.py

# Test multi-format support
python test_features.py

# Test environment configuration
python test_env_config.py
```

### Adding Tests

When adding new features:

1. **Add test cases** for new functionality
2. **Ensure existing tests pass** after your changes
3. **Test edge cases** and error conditions
4. **Test with different document formats** if applicable

### Test Structure

```python
def test_new_feature():
    """Test description explaining what is being tested."""
    # Setup
    test_input = "example input"
    
    # Execute
    result = your_function(test_input)
    
    # Assert
    assert result == expected_output
    print("Test passed: new feature works correctly")
```

## Feature Areas

### Current Architecture

The project consists of several key modules:

- **`src/search.py`**: Hybrid search combining semantic and keyword approaches
- **`src/llm_router.py`**: LLM provider abstraction layer
- **`src/conversation.py`**: Conversational AI with memory management
- **`src/summarization.py`**: Document summarization engine
- **`src/ingestion.py`**: Multi-format document loading
- **`src/preprocessing.py`**: Text processing pipeline
- **`src/indexing.py`**: Incremental indexing system
- **`src/ocr.py`**: OCR processing for scanned documents

### Areas for Contribution

1. **New Document Formats**: Add support for additional file types
2. **Enhanced OCR**: Improve OCR accuracy and performance
3. **Search Improvements**: Better ranking algorithms, query understanding
4. **LLM Integration**: Support for new LLM providers
5. **User Interface**: Web interface, improved CLI experience
6. **Performance**: Optimization for large document collections
7. **Deployment**: Docker, cloud deployment scripts
8. **Monitoring**: Logging, metrics, health checks

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation** as needed
3. **Check code style** compliance
4. **Verify no sensitive data** is included (API keys, personal documents)

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added/updated
- [ ] README/USAGE updated if needed
- [ ] ROADMAP updated if adding major features
```

## Code Review Process

1. **Automated checks** will run on your PR
2. **Maintainers will review** your code and provide feedback
3. **Address feedback** by making additional commits
4. **Once approved**, your PR will be merged

## Community Guidelines

### Be Respectful

- Use inclusive language
- Be constructive in feedback
- Help newcomers get started
- Respect different perspectives and approaches

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Pull Requests**: Keep discussion focused on the specific changes

## Getting Help

- **Documentation**: Check [USAGE.md](USAGE.md) and [README.md](README.md)
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub discussions for general questions
- **Code**: Look at existing code for patterns and examples

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project documentation and release notes. Thank you for helping make this project better!