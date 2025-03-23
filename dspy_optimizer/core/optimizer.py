import os
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import dspy
from typing import Dict, List, Optional, Any

from dspy_optimizer.utils.metrics import linkedin_style_metric, linkedin_content_metric
from dspy_optimizer.core.modules import LinkedInStyleAnalyzer, LinkedInContentTransformer

def configure_lm():
    load_dotenv()
    
    if os.getenv("GEMINI_API_KEY"):
        try:
            return dspy.LM(
                model="openai/gemini-1.5-flash",
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv("GEMINI_API_KEY")
            )
        except Exception as e:
            logging.warning(f"Failed to initialize Gemini model: {e}")
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            return dspy.LM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=15
            )
        except Exception as e:
            logging.warning(f"Failed to initialize GPT-4o-mini: {e}")
            try:
                return dspy.LM(
                    model="openai/gpt-3.5-turbo",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=25
                )
            except Exception as e:
                logging.warning(f"Failed to initialize GPT-3.5-turbo: {e}")
    
    logging.warning("No API keys found or all model initializations failed. Using default LM.")
    return dspy.LM()

def run_optimization(module, trainset, metric, output_dir, filename, module_name):
    if len(trainset) < 2:
        logging.warning(f"Not enough examples for {module_name} optimization, returning unoptimized {module_name}")
        return module
    
    optimizer = dspy.MIPROv2(
        metric=metric,
        max_bootstrapped_demos=3,
        num_candidates=5,
        auto="light"
    )
    
    try:
        optimized_module = optimizer.compile(
            module, 
            trainset=trainset,
            num_trials=5,
            requires_permission_to_run=False
        )
        
        save_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        optimized_module.save(str(save_path))
        
        return optimized_module
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            logging.error(f"{module_name.capitalize()} rate limit exceeded: {e}")
            logging.info("Waiting for 60 seconds before retrying...")
            time.sleep(60)
            try:
                optimized_module = optimizer.compile(
                    module, 
                    trainset=trainset,
                    num_trials=3,
                    requires_permission_to_run=False
                )
                
                save_path = Path(output_dir) / filename
                optimized_module.save(str(save_path))
                
                return optimized_module
            except Exception as retry_e:
                logging.error(f"{module_name.capitalize()} optimization retry failed: {retry_e}")
        else:
            logging.error(f"{module_name.capitalize()} optimization error: {e}")
        
        return module

def optimize_analyzer(train_examples, output_dir="."):
    return run_optimization(
        module=LinkedInStyleAnalyzer(),
        trainset=train_examples,
        metric=linkedin_style_metric,
        output_dir=output_dir,
        filename="optimized_linkedin_analyzer.json",
        module_name="analyzer"
    )

def extract_prompt_from_module(module):
    default_prompts = {
        LinkedInStyleAnalyzer: "You are an AI that analyzes LinkedIn content to identify effective posting strategies. Extract key style characteristics such as tone, structure, emoji usage, hooks, calls-to-action, and formatting.",
        LinkedInContentTransformer: "You are a LinkedIn content expert. Transform the provided content into an engaging LinkedIn post by applying the style characteristics. Focus on creating attention-grabbing hooks, using emojis strategically, formatting for readability, and adding appropriate hashtags."
    }
    
    module_type = type(module)
    default_prompt = default_prompts.get(module_type, f"{module.__class__.__name__} optimized prompt")
    
    try:
        attrs_to_check = [
            ('_compiled_lm', 'program_template'),
            ('analyzer', 'template'),
            ('transformer', 'template'),
            ('predictor', 'template')
        ]
        
        for attr, subattr in attrs_to_check:
            if hasattr(module, attr):
                obj = getattr(module, attr)
                if hasattr(obj, subattr):
                    prompt = getattr(obj, subattr)
                    if prompt and isinstance(prompt, str):
                        return prompt
    except Exception as e:
        logging.warning(f"Error extracting prompt from module: {e}")
        
    return default_prompt

def optimize_transformer(train_examples, analyzer, output_dir="."):
    return run_optimization(
        module=LinkedInContentTransformer(),
        trainset=train_examples,
        metric=linkedin_content_metric,
        output_dir=output_dir,
        filename="optimized_linkedin_transformer.json",
        module_name="transformer"
    )

def extract_optimized_prompts(output_dir="."):
    output_path = Path(output_dir)
    analyzer_path = output_path / "optimized_linkedin_analyzer.json"
    transformer_path = output_path / "optimized_linkedin_transformer.json"
    
    analyzer = LinkedInStyleAnalyzer()
    transformer = LinkedInContentTransformer()
    
    if analyzer_path.exists():
        try:
            analyzer.load(str(analyzer_path))
        except Exception as e:
            logging.warning(f"Error loading analyzer from {analyzer_path}: {e}")
    
    if transformer_path.exists():
        try:
            transformer.load(str(transformer_path))
        except Exception as e:
            logging.warning(f"Error loading transformer from {transformer_path}: {e}")
    
    prompts = {
        "linkedin_analyzer_prompt": extract_prompt_from_module(analyzer),
        "linkedin_transformer_prompt": extract_prompt_from_module(transformer)
    }
    
    prompts_path = output_path / "optimized_prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)
    
    return prompts