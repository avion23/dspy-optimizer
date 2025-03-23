import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional

import dspy
from dspy_optimizer.utils.data import load_examples, prepare_datasets
from dspy_optimizer.core.optimizer import (
    configure_lm, 
    optimize_extractor, 
    optimize_applicator,
    extract_optimized_prompts
)

def optimize(examples_path: Optional[str] = None, output_dir: str = ".") -> Dict[str, str]:
    load_dotenv()
    
    try:
        lm = configure_lm()
        dspy.settings.configure(lm=lm)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        examples = load_examples(examples_path)
        train_extractor_examples, train_examples, test_examples = prepare_datasets(examples)
        
        optimized_extractor = optimize_extractor(train_extractor_examples, output_dir)
        optimized_applicator = optimize_applicator(train_examples, optimized_extractor, output_dir)
        
        return extract_optimized_prompts(output_dir)
    except Exception as e:
        return {"error": str(e)}

def apply_to_app(app_path: Optional[str] = None, prompts_path: str = "optimized_prompts.json", dry_run: bool = False) -> bool:
    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        return False
    
    try:
        with open(prompts_file, "r") as f:
            prompts = json.load(f)
    except json.JSONDecodeError:
        return False
    
    required_keys = ["style_analyzer_prompt", "style_applicator_prompt"]
    if not all(key in prompts for key in required_keys):
        return False
        
    app_path = Path(app_path or os.path.expanduser("~/Documents/projects/app_src"))
    
    style_page_path = app_path / "lib" / "style_page.dart"
    chat_page_path = app_path / "lib" / "chat_page.dart"
    
    update_successful = False
    
    if style_page_path.exists():
        try:
            with open(style_page_path, "r") as f:
                content = f.read()
            
            old_analyzer_prompt = "You are an AI that analyzes text and creates concise style instructions for other AI's to rewrite texts."
            if old_analyzer_prompt in content:
                new_content = content.replace(old_analyzer_prompt, prompts["style_analyzer_prompt"])
                
                if not dry_run:
                    with open(style_page_path, "w") as f:
                        f.write(new_content)
                update_successful = True
        except Exception:
            pass
    
    if chat_page_path.exists():
        try:
            with open(chat_page_path, "r") as f:
                content = f.read()
            
            old_applicator_prompt = "You are a helpful assistant for writing any sort of texts."
            if old_applicator_prompt in content:
                new_content = content.replace(old_applicator_prompt, prompts["style_applicator_prompt"])
                
                if not dry_run:
                    with open(chat_page_path, "w") as f:
                        f.write(new_content)
                update_successful = True
        except Exception:
            pass
        
    return update_successful

def main():
    parser = argparse.ArgumentParser(description="DSPy Style Prompt Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    optimize_parser = subparsers.add_parser("optimize", help="Run optimization")
    optimize_parser.add_argument(
        "--examples", "-e", 
        help="Path to examples file (optional)"
    )
    optimize_parser.add_argument(
        "--output", "-o", 
        default=".", 
        help="Directory to save optimized models and prompts"
    )
    
    apply_parser = subparsers.add_parser("apply", help="Apply optimized prompts to an application")
    apply_parser.add_argument(
        "app_path", 
        nargs="?", 
        help="Path to application directory"
    )
    apply_parser.add_argument(
        "--prompts", "-p", 
        default="optimized_prompts.json", 
        help="Path to optimized prompts JSON file"
    )
    apply_parser.add_argument(
        "--dry-run", "-d", 
        action="store_true", 
        help="Preview changes without writing"
    )
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        optimize(args.examples, args.output)
    elif args.command == "apply":
        apply_to_app(args.app_path, args.prompts, args.dry_run)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()