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

def prepare_datasets(examples) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    # Always apply 70/30 split, but ensure at least one example for test
    split_index = max(1, int(len(examples) * 0.7))
    # For very small datasets (2 examples), force a 1/1 split
    if len(examples) == 2:
        split_index = 1
    
    train_extractor_examples = []
    for ex in examples[:split_index]:
        example = dspy.Example(
            sample_text=ex["sample"],
            content_to_style=ex["content_to_style"],
            expected_styled_content=ex["expected_styled_content"]
        )
        # Use the actual inputs property, not the method
        example = example.with_inputs("sample_text")
        train_extractor_examples.append(example)
    
    train_examples = []
    for ex in examples[:split_index]:
        example = dspy.Example(
            sample_text=ex["sample"],
            content_to_style=ex["content_to_style"],
            expected_styled_content=ex["expected_styled_content"]
        )
        example = example.with_inputs("sample_text", "content_to_style")
        train_examples.append(example)
    
    test_examples = []
    for ex in examples[split_index:]:
        example = dspy.Example(
            sample_text=ex["sample"],
            content_to_style=ex["content_to_style"],
            expected_styled_content=ex["expected_styled_content"]
        )
        example = example.with_inputs("sample_text", "content_to_style")
        test_examples.append(example)
    
    return train_extractor_examples, train_examples, test_examples