import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import dspy

from dspy_optimizer.utils.data import load_examples, prepare_datasets
from dspy_optimizer.core.optimizer import (
    configure_lm, 
    optimize_analyzer, 
    optimize_transformer,
    extract_optimized_prompts
)

def optimize(examples_path: Optional[str] = None, output_dir: str = ".") -> Dict:
    try:
        # Configure language model
        lm = configure_lm()
        dspy.settings.configure(lm=lm)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default to LinkedIn examples if no path specified
        if examples_path is None:
            examples_path = os.path.join(os.path.dirname(__file__), "..", "linkedin_examples.json")
        
        examples = load_examples(examples_path)
        train_analyzer_examples, train_examples, test_examples = prepare_datasets(examples)
        
        optimized_analyzer = optimize_analyzer(train_analyzer_examples, output_dir)
        optimized_transformer = optimize_transformer(train_examples, optimized_analyzer, output_dir)
        
        return extract_optimized_prompts(output_dir)
    except Exception as e:
        print(f"Optimization error: {e}")
        return {"error": str(e)}

def apply_to_app(app_path: Optional[str] = None, prompts_path: str = "optimized_prompts.json", dry_run: bool = False) -> bool:
    try:
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"Prompts file not found: {prompts_path}")
        return False
    except json.JSONDecodeError:
        print(f"Invalid JSON in prompts file: {prompts_path}")
        return False
    
    # Check if we have LinkedIn prompts
    linkedin_prompts = "linkedin_analyzer_prompt" in prompts and "linkedin_transformer_prompt" in prompts
    
    app_path = Path(app_path or os.path.expanduser("~/Documents/projects/app_src"))
    
    update_successful = False
    
    # Check for social media content app files
    social_page_path = app_path / "lib" / "social_media_content.dart"
    linkedin_page_path = app_path / "lib" / "linkedin_post.dart"
    
    if linkedin_prompts and (social_page_path.exists() or linkedin_page_path.exists()):
        try:
            target_path = linkedin_page_path if linkedin_page_path.exists() else social_page_path
            with open(target_path, "r") as f:
                content = f.read()
            
            # Update analyzer prompt
            old_analyzer_prompt = "You are an AI that analyzes LinkedIn posts."
            if old_analyzer_prompt in content:
                new_content = content.replace(old_analyzer_prompt, prompts["linkedin_analyzer_prompt"])
                
                # Update transformer prompt
                old_transformer_prompt = "Transform the content into an engaging LinkedIn post."
                if old_transformer_prompt in new_content:
                    new_content = new_content.replace(old_transformer_prompt, prompts["linkedin_transformer_prompt"])
                
                if not dry_run:
                    with open(target_path, "w") as f:
                        f.write(new_content)
                update_successful = True
                
                print(f"Updated LinkedIn prompts in {target_path}")
        except Exception as e:
            print(f"Error updating app files: {e}")
    
    return update_successful

def main():
    parser = argparse.ArgumentParser(description="DSPy LinkedIn Content Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    optimize_parser = subparsers.add_parser("optimize", help="Run optimization")
    optimize_parser.add_argument(
        "--examples", "-e", 
        help="Path to examples file (defaults to linkedin_examples.json)"
    )
    optimize_parser.add_argument(
        "--output", "-o", 
        default="./output", 
        help="Directory to save optimized models and prompts"
    )
    
    apply_parser = subparsers.add_parser("apply", help="Apply optimized prompts to app")
    apply_parser.add_argument(
        "--app-path", "-a", 
        help="Path to app source code (optional)"
    )
    apply_parser.add_argument(
        "--prompts", "-p", 
        default="output/optimized_prompts.json", 
        help="Path to optimized prompts JSON file"
    )
    apply_parser.add_argument(
        "--dry-run", "-d", 
        action="store_true", 
        help="Preview changes without writing to files"
    )
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        optimize(args.examples, args.output)
        print(f"Optimization complete. Results saved to {args.output}")
    elif args.command == "apply":
        success = apply_to_app(args.app_path, args.prompts, args.dry_run)
        if success:
            print("Successfully applied optimized prompts to application.")
        else:
            print("Failed to apply prompts. Make sure the prompt file exists and application paths are correct.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()