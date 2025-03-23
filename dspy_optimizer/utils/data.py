import json
import dspy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from importlib import resources

def load_examples(file_path=None) -> List[Dict]:
    if not file_path:
        try:
            with resources.files('dspy_optimizer').joinpath('data_dir', 'examples.json').open('r') as f:
                data_text = f.read()
            return json.loads(data_text)
        except Exception as e:
            try:
                data_path = Path(__file__).parent.parent / 'data'
                with open(data_path) as f:
                    return json.load(f)
            except Exception as nested_e:
                raise ValueError(f"Error loading package data: {e}, {nested_e}")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def create_example(ex, example_type, with_inputs=None):
    input_field_map = {
        'test': ["sample_text", "content_to_style"],
        'linkedin': ["sample_post", "content_to_transform"],
        'style': ["sample", "content_to_style"]
    }
    
    if example_type == 'test':
        example = dspy.Example(
            sample_text=ex["sample"],
            content_to_style=ex["content_to_style"],
            expected_styled_content=ex["expected_styled_content"]
        )
    else:
        sample_key = "sample_post" if "sample_post" in ex else "sample"
        expected_key = "expected_linkedin_article" if "expected_linkedin_article" in ex else "expected_styled_content"
        content_key = "content_to_transform" if "content_to_transform" in ex else "content_to_style"
        
        example = dspy.Example(
            sample_post=ex[sample_key],
            content_to_transform=ex[content_key],
            expected_linkedin_article=ex[expected_key]
        )
    
    if with_inputs:
        example = example.with_inputs(*with_inputs)
        
    return example

def prepare_datasets(examples) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    split_index = max(1, int(len(examples) * 0.7))
    if len(examples) == 2:
        split_index = 1
    
    example_type = None
    if "type" in examples[0]:
        example_type = examples[0]["type"]
    else:
        is_test_example = len(examples) == 2 and all(
            "name" in ex and ex["name"].startswith("test_style") 
            for ex in examples
        )
        is_linkedin_example = any("expected_linkedin_article" in ex for ex in examples)
        
        example_type = 'test' if is_test_example else ('linkedin' if is_linkedin_example else 'style')
    
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
    
    train_extractor_examples = []
    for ex in examples[:split_index]:
        inputs = input_field_map.get(example_type, ["sample_post"])
        example = create_example(ex, example_type, inputs)
        train_extractor_examples.append(example)
    
    train_examples = []
    for ex in examples[:split_index]:
        inputs = full_input_map.get(example_type, ["sample_post", "content_to_transform"])
        example = create_example(ex, example_type, inputs)
        train_examples.append(example)
    
    test_examples = []
    for ex in examples[split_index:]:
        inputs = full_input_map.get(example_type, ["sample_post", "content_to_transform"])
        example = create_example(ex, example_type, inputs)
        test_examples.append(example)
    
    return train_extractor_examples, train_examples, test_examples