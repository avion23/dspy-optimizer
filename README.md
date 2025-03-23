# DSPy Style Prompt Optimizer

A streamlined framework for optimizing style extraction and application prompts using DSPy.

## Overview

This package optimizes language model prompts for style-related tasks:

1. **Style Extraction**: Analyzes text to identify stylistic characteristics
2. **Style Application**: Applies identified style to new content
3. **Evaluation**: Measures style transfer effectiveness

## Quick Start

```bash
# Install package
pip install .

# Create .env file with API keys
echo "OPENAI_API_KEY=your_key_here" > .env
# OR
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Project Structure

```
dspy_optimizer/
├── core/           # Core modules for style extraction and application
├── utils/          # Data handling utilities
├── tests/          # Test suite
├── data            # Example style data
└── cli.py          # Command line interface
```

## Command Line Usage

```bash
# Run optimization with default examples
dspy-optimize

# Run with custom examples
dspy-optimize --examples path/to/examples.json --output ./output

# Apply optimized prompts to an application
dspy-apply path/to/app/directory

# Preview changes without applying
dspy-apply path/to/app/directory --dry-run
```

## Example Format

Examples should be provided in JSON format:

```json
[
  {
    "name": "formal_academic",
    "sample": "The results indicate a statistically significant correlation between variables X and Y (p < 0.01).",
    "content_to_style": "X and Y are related.",
    "expected_styled_content": "The data suggests a correlation between variables X and Y."
  }
]
```

## Features

- Optimizes style prompts using DSPy's MIPROv2
- Supports both OpenAI and Google Gemini models
- Includes metrics for style quality and application
- Provides command-line interface for easy usage
- Test-driven development with comprehensive test coverage