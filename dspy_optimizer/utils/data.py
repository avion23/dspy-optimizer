import json
import dspy
from pathlib import Path
from typing import Dict, List, Tuple
from importlib import resources

def load_examples(file_path=None):
    if not file_path:
        default_paths = [
            Path(__file__).parent.parent.parent / 'linkedin_examples.json',
            Path.cwd() / 'linkedin_examples.json'
        ]
        for path in default_paths:
            if path.exists():
                file_path = path
                break
        if not file_path:
            raise FileNotFoundError("No examples file found and no default file path provided")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def create_example(ex, example_type, with_inputs=None):
    field_mappings = {
        'test': {
            'sample_field': 'sample',
            'content_field': 'content_to_style',
            'expected_field': 'expected_styled_content'
        },
        'linkedin': {
            'sample_field': 'sample_post' if 'sample_post' in ex else 'sample',
            'content_field': 'content_to_transform',
            'expected_field': 'expected_linkedin_article'
        },
        'style': {
            'sample_field': 'sample',
            'content_field': 'content_to_style',
            'expected_field': 'expected_styled_content'
        }
    }
    
    mapping = field_mappings.get(example_type, field_mappings['linkedin'])
    sample_key = mapping['sample_field']
    content_key = mapping['content_field']
    expected_key = mapping['expected_field']
    
    if example_type == 'test':
        example = dspy.Example(
            sample_text=ex[sample_key],
            content_to_style=ex[content_key],
            expected_styled_content=ex[expected_key]
        )
    else:
        example = dspy.Example(
            sample_post=ex[sample_key],
            content_to_transform=ex[content_key],
            expected_linkedin_article=ex[expected_key]
        )
    
    if with_inputs:
        example = example.with_inputs(*with_inputs)
        
    return example

def detect_example_type(examples):
    if not examples:
        raise ValueError("Empty examples list provided")
        
    if "type" in examples[0]:
        return examples[0]["type"]
    
    is_test = all("name" in ex and ex["name"].startswith("test_style") for ex in examples)
    has_linkedin = any("expected_linkedin_article" in ex for ex in examples)
    
    return 'test' if is_test else ('linkedin' if has_linkedin else 'style')

def get_input_fields(example_type):
    input_field_map = {
        'test': ["sample_text"],
        'linkedin': ["sample_post"],
        'style': ["sample"]
    }
    
    full_input_map = {
        'test': ["sample_text", "content_to_style"],
        'linkedin': ["sample_post", "content_to_transform"],
        'style': ["sample", "content_to_style"]
    }
    
    return input_field_map.get(example_type, ["sample_post"]), full_input_map.get(example_type, ["sample_post", "content_to_transform"])

def prepare_datasets(examples):
    if not examples:
        raise ValueError("Empty examples list provided")
        
    split_index = max(1, int(len(examples) * 0.7))
    if len(examples) == 2:
        split_index = 1
    
    example_type = detect_example_type(examples)
    single_inputs, full_inputs = get_input_fields(example_type)
    
    train_extractor_examples = [
        create_example(ex, example_type, single_inputs) 
        for ex in examples[:split_index]
    ]
    
    train_examples = [
        create_example(ex, example_type, full_inputs) 
        for ex in examples[:split_index]
    ]
    
    test_examples = [
        create_example(ex, example_type, full_inputs) 
        for ex in examples[split_index:]
    ]
    
    return train_extractor_examples, train_examples, test_examples