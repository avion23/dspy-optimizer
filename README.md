# DSPy LinkedIn Content Optimizer

A DSPy-based tool for optimizing LinkedIn content creation. This tool analyzes LinkedIn posts to extract style characteristics and transforms plain content into engaging LinkedIn posts that match the extracted style.

## Features

- **LinkedIn Style Analysis**: Extracts style characteristics from sample LinkedIn posts
- **Content Transformation**: Transforms plain content into engaging LinkedIn posts
- **Quality Evaluation**: Evaluates the quality of generated content
- **Prompt Optimization**: Uses DSPy's MIPROv2 to optimize prompts

## Installation

```bash
# Install dependencies
pip install -e .
```

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
# or
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### CLI

```bash
# Run optimization using LinkedIn examples
python -m dspy_optimizer.cli optimize --examples linkedin_examples.json --output ./output

# Apply optimized prompts to an app
python -m dspy_optimizer.cli apply --app-path /path/to/app --prompts ./output/optimized_prompts.json
```

### Demo

```bash
python demo_linkedin_optimizer.py
```

## Example

**Input Content**:
```
Our company research found that remote employees are more productive but face communication challenges. 
We surveyed 500 remote workers and found they complete 22% more tasks but spend 3.2 hours daily on 
communication tools. Asynchronous communication methods can help reduce meeting time.
```

**Generated LinkedIn Post**:
```
**Remote Work: A Productivity Paradox?** 🔍

Our recent research reveals a stunning insight: remote workers are 22% MORE productive, but at what cost?

The hidden challenge? They're spending 3.2 hours DAILY navigating communication tools!

How can we optimize this process?

✅ Implement asynchronous communication
✅ Establish clear documentation practices
✅ Use project management tools effectively

Is your team struggling with similar challenges? Share your experience below! 👇

#RemoteWork #Productivity #WorkFromHome
```

## Project Structure

- `dspy_optimizer/`: Main package
  - `core/`: Core modules and optimization logic
  - `utils/`: Utility functions
- `linkedin_examples.json`: LinkedIn examples for training
- `demo_linkedin_optimizer.py`: Demo script

## License

MIT