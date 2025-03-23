import sys
import os

print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    import dspy
    print("DSPy version:", dspy.__version__)
except ImportError:
    print("DSPy not found")

try:
    import litellm
    print("LiteLLM version:", litellm.__version__)
except ImportError:
    print("LiteLLM not found")
