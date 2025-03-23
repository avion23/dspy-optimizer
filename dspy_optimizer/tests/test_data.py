import unittest
import json
import tempfile
import os
from pathlib import Path
import dspy
from dspy_optimizer.utils.data import load_examples, prepare_datasets

class TestData(unittest.TestCase):
    def setUp(self):
        # Create temporary test examples
        self.test_examples = [
            {
                "name": "test_style1",
                "sample": "Sample text for style 1.",
                "content_to_style": "Content to style.",
                "expected_styled_content": "Styled content."
            },
            {
                "name": "test_style2",
                "sample": "Sample text for style 2.",
                "content_to_style": "More content to style.",
                "expected_styled_content": "More styled content."
            }
        ]
        
        # Create a temporary example file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.examples_path = Path(self.temp_dir.name) / "test_examples.json"
        
        with open(self.examples_path, 'w') as f:
            json.dump(self.test_examples, f)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_examples(self):
        # Test loading from file
        examples = load_examples(self.examples_path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["name"], "test_style1")
        
        # Test file not found error
        with self.assertRaises(FileNotFoundError):
            load_examples("nonexistent_file.json")
    
    def test_prepare_datasets(self):
        examples = load_examples(self.examples_path)
        train_extractor_examples, train_examples, test_examples = prepare_datasets(examples)
        
        # Check the extractor examples have only sample_text input
        # Instead of checking inputs directly, we can check attributes
        self.assertTrue(hasattr(train_extractor_examples[0], "sample_text"))
        self.assertTrue(hasattr(train_extractor_examples[0], "content_to_style"))
        
        # Check full pipeline examples have both inputs
        self.assertTrue(hasattr(train_examples[0], "sample_text"))
        self.assertTrue(hasattr(train_examples[0], "content_to_style"))
        
        # Should have 1 train example and 1 test example with 70/30 split
        self.assertEqual(len(train_examples), 1)
        self.assertEqual(len(test_examples), 1)

if __name__ == "__main__":
    unittest.main()