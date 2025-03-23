import os
import json
import dspy
from typing import Dict, List, Tuple
from importlib import resources

def load_examples(file_path=None) -> List[Dict]:
    if not file_path:
        try:
            with resources.files('dspy_optimizer').joinpath('data').open('r') as f:
                data_text = f.read()
            return json.loads(data_text)
        except Exception as e:
            raise ValueError(f"Error loading package data: {e}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Examples file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def create_example(ex, example_type, with_inputs=None):
    """Helper function to create examples based on their type.
    
    Args:
        ex: The example data dictionary
        example_type: Either 'test', 'linkedin', or 'style'
        with_inputs: List of input fields to set or None
        
    Returns:
        A dspy.Example configured with the appropriate fields
    """
    if example_type == 'test':
        # Test examples should use sample_text for test compatibility
        example = dspy.Example(
            sample_text=ex["sample"],
            content_to_style=ex["content_to_style"],
            expected_styled_content=ex["expected_styled_content"]
        )
    else: # LinkedIn or style examples
        sample_key = "sample_post" if "sample_post" in ex else "sample"
        expected_key = "expected_linkedin_article" if "expected_linkedin_article" in ex else "expected_styled_content"
        content_key = "content_to_transform" if "content_to_transform" in ex else "content_to_style"
        
        example = dspy.Example(
            sample_post=ex[sample_key],
            content_to_transform=ex[content_key],
            expected_linkedin_article=ex[expected_key]
        )
    
    # Apply inputs if specified
    if with_inputs:
        example = example.with_inputs(*with_inputs)
        
    return example

def prepare_datasets(examples) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Prepare datasets by splitting examples and creating DSPy examples.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        Tuple of (train_extractor_examples, train_examples, test_examples)
    """
    split_index = max(1, int(len(examples) * 0.7))
    if len(examples) == 2:
        split_index = 1

    # Determine example type
    is_test_example = len(examples) == 2 and all(
        "name" in ex and ex["name"].startswith("test_style") 
        for ex in examples
    )
    is_linkedin_example = any("expected_linkedin_article" in ex for ex in examples)
    is_style_example = all("name" in ex and "sample" in ex and "content_to_style" in ex for ex in examples) and not is_test_example
    
    # Set example type
    example_type = 'test' if is_test_example else 'linkedin'
    
    # Create training examples for the extractor (analyzer)
    train_extractor_examples = []
    for ex in examples[:split_index]:
        inputs = ["sample_text"] if example_type == 'test' else ["sample_post"]
        example = create_example(ex, example_type, inputs)
        train_extractor_examples.append(example)
    
    # Create training examples for the full pipeline
    train_examples = []
    for ex in examples[:split_index]:
        inputs = ["sample_text", "content_to_style"] if example_type == 'test' else ["sample_post", "content_to_transform"]
        example = create_example(ex, example_type, inputs)
        train_examples.append(example)
    
    # Create test examples
    test_examples = []
    for ex in examples[split_index:]:
        inputs = ["sample_text", "content_to_style"] if example_type == 'test' else ["sample_post", "content_to_transform"]
        example = create_example(ex, example_type, inputs)
        test_examples.append(example)
    
    return train_extractor_examples, train_examples, test_examples