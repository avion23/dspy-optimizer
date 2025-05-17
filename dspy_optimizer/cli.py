import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from time import time
from tabulate import tabulate

import dspy

from dspy_optimizer.utils.data import load_examples, prepare_datasets
from dspy_optimizer.core.optimizer import (
    configure_lm, 
    optimize_analyzer, 
    optimize_transformer,
    extract_optimized_prompts
)

def optimize(examples_path=None, output_dir=".", trials=7, candidates=5, bootstrapped_demos=3):
    try:
        start_time = time()
        
        lm = configure_lm()
        dspy.settings.configure(lm=lm)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if examples_path is None:
            examples_path = os.path.join(os.path.dirname(__file__), "..", "linkedin_examples.json")
        
        logging.info(f"Optimizing LinkedIn content using examples from: {examples_path}")
        logging.info(f"Parameters: {trials} trials, {candidates} candidates, {bootstrapped_demos} bootstrapped demos")
        
        examples = load_examples(examples_path)
        train_analyzer_examples, train_examples, _ = prepare_datasets(examples)
        
        logging.info(f"Loaded {len(examples)} examples, split into {len(train_analyzer_examples)} for analyzer training")
        
        logging.info("Optimizing LinkedIn Style Analyzer...")
        optimized_analyzer = optimize_analyzer(train_analyzer_examples, output_dir, trials, bootstrapped_demos, candidates)
        
        logging.info("Optimizing LinkedIn Content Transformer...")
        optimized_transformer = optimize_transformer(train_examples, optimized_analyzer, output_dir, trials, bootstrapped_demos, candidates)
        
        prompts = extract_optimized_prompts(output_dir)
        
        end_time = time()
        duration = end_time - start_time
        
        logging.info("Optimization Results Summary:")
        results = [
            ["Analyzer Prompt", f"{len(prompts['linkedin_analyzer_prompt'])} chars"],
            ["Transformer Prompt", f"{len(prompts['linkedin_transformer_prompt'])} chars"],
            ["Duration", f"{duration:.1f} seconds"],
            ["Output Directory", output_dir]
        ]
        logging.info("\n" + tabulate(results, tablefmt="simple"))
        
        return prompts
    except Exception as e:
        logging.error(f"Optimization error: {e}")
        return {"error": str(e)}

def apply_to_app(app_path=None, prompts_path="optimized_prompts.json", dry_run=False):
    try:
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        logging.error(f"Prompts file not found: {prompts_path}")
        return False
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in prompts file: {prompts_path}")
        return False
    
    if not app_path:
        app_path = input("Enter the application directory path (e.g., ./app): ")
        if not app_path:
            logging.error("No application path provided")
            return False
    
    app_path = Path(app_path)
    if not app_path.exists():
        logging.error(f"Application path not found: {app_path}")
        return False
    
    linkedin_prompts = "linkedin_analyzer_prompt" in prompts and "linkedin_transformer_prompt" in prompts
    if not linkedin_prompts:
        logging.error("Required prompts not found in prompts file")
        return False
    
    target_files = [
        app_path / "lib" / "linkedin_post.dart",
        app_path / "lib" / "social_media_content.dart",
        app_path / "lib" / "src" / "linkedin_post.dart",
        app_path / "lib" / "src" / "social_media_content.dart"
    ]
    
    target_path = None
    for path in target_files:
        if path.exists():
            target_path = path
            break
    
    if not target_path:
        logging.error(f"Could not find any target file in {app_path}")
        return False
    
    try:
        with open(target_path, "r") as f:
            content = f.read()
        
        analyzer_start = "// DSPY_ANALYZER_PROMPT_START"
        analyzer_end = "// DSPY_ANALYZER_PROMPT_END"
        transformer_start = "// DSPY_TRANSFORMER_PROMPT_START"
        transformer_end = "// DSPY_TRANSFORMER_PROMPT_END"
        
        if analyzer_start in content and analyzer_end in content and transformer_start in content and transformer_end in content:
            analyzer_pattern = f"{analyzer_start}(.*?){analyzer_end}"
            transformer_pattern = f"{transformer_start}(.*?){transformer_end}"
            
            new_content = re.sub(
                analyzer_pattern, 
                f"{analyzer_start}{prompts['linkedin_analyzer_prompt']}{analyzer_end}", 
                content, 
                flags=re.DOTALL
            )
            new_content = re.sub(
                transformer_pattern, 
                f"{transformer_start}{prompts['linkedin_transformer_prompt']}{transformer_end}", 
                new_content, 
                flags=re.DOTALL
            )
        else:
            old_analyzer_prompt = "You are an AI that analyzes LinkedIn posts."
            old_transformer_prompt = "Transform the content into an engaging LinkedIn post."
            
            if old_analyzer_prompt not in content or old_transformer_prompt not in content:
                logging.warning("Could not find placeholder markers or target prompts in application file")
                logging.warning("Falling back to exact string match replacement")
                
            new_content = content.replace(old_analyzer_prompt, prompts["linkedin_analyzer_prompt"])
            new_content = new_content.replace(old_transformer_prompt, prompts["linkedin_transformer_prompt"])
        
        if content == new_content:
            logging.warning("No changes were made to the file content")
            return False
        
        if dry_run:
            logging.info(f"Dry run: Would update prompts in {target_path}")
        else:
            with open(target_path, "w") as f:
                f.write(new_content)
            logging.info(f"Updated LinkedIn prompts in {target_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error updating app files: {e}")
        return False

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
    optimize_parser.add_argument(
        "--trials", "-t",
        type=int,
        default=5,
        help="Number of optimization trials (default: 5)"
    )
    optimize_parser.add_argument(
        "--candidates", "-c",
        type=int,
        default=5,
        help="Number of prompt candidates (default: 5)"
    )
    optimize_parser.add_argument(
        "--demos", "-d",
        type=int,
        default=3,
        help="Maximum number of bootstrapped demonstrations (default: 3)"
    )
    
    apply_parser = subparsers.add_parser("apply", help="Apply optimized prompts to app")
    apply_parser.add_argument(
        "--app-path", "-a", 
        help="Path to app source code"
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
        optimize(args.examples, args.output, args.trials, args.candidates, args.demos)
        logging.info(f"Optimization complete. Results saved to {args.output}")
    elif args.command == "apply":
        success = apply_to_app(args.app_path, args.prompts, args.dry_run)
        if success:
            logging.info("Successfully applied optimized prompts to application.")
        else:
            logging.error("Failed to apply prompts. Check logs for details.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()