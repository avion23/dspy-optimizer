import os
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import dspy

from dspy_optimizer.utils.metrics import linkedin_style_metric, linkedin_content_metric
from dspy_optimizer.core.modules import LinkedInStyleAnalyzer, LinkedInContentTransformer

def configure_lm():
    load_dotenv()
    
    models = [
        ("GEMINI_API_KEY", lambda: dspy.LM(
            model="openai/gemini-1.5-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY")
        )),
        ("OPENAI_API_KEY", lambda: dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=15
        )),
    ]
    
    for env_var, model_init in models:
        if not os.getenv(env_var):
            continue
            
        try:
            return model_init()
        except Exception as e:
            logging.warning(f"Failed to initialize model with {env_var}: {e}")
    
    logging.warning("No API keys found or all model initializations failed. Using default LM.")
    return dspy.LM()

def run_optimization(module, trainset, metric, output_dir, filename, module_name, num_trials=5, num_bootstrapped_demos=3, num_candidates=5):
    if len(trainset) < 2:
        logging.warning(f"Not enough examples for {module_name} optimization (minimum 2 required), returning unoptimized {module_name}")
        return module
    
    optimizer = dspy.MIPROv2(
        metric=metric,
        max_bootstrapped_demos=num_bootstrapped_demos,
        num_candidates=num_candidates,
        auto="light"
    )
    
    try:
        logging.info(f"Starting {module_name} optimization with {len(trainset)} examples")
        optimized_module = optimizer.compile(
            module, 
            trainset=trainset,
            num_trials=num_trials,
            requires_permission_to_run=False
        )
        
        save_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        optimized_module.save(str(save_path))
        logging.info(f"Saved optimized {module_name} to {save_path}")
        
        return optimized_module
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            logging.error(f"{module_name.capitalize()} rate limit exceeded: {e}")
            logging.info("Waiting for 60 seconds before retrying...")
            time.sleep(60)
            try:
                reduced_trials = max(2, num_trials // 2)
                logging.info(f"Retrying {module_name} optimization with reduced trials ({reduced_trials})")
                optimized_module = optimizer.compile(
                    module, 
                    trainset=trainset,
                    num_trials=reduced_trials,
                    requires_permission_to_run=False
                )
                
                save_path = Path(output_dir) / filename
                optimized_module.save(str(save_path))
                logging.info(f"Saved optimized {module_name} to {save_path} after retry")
                
                return optimized_module
            except Exception as retry_e:
                logging.error(f"{module_name.capitalize()} optimization retry failed: {retry_e}")
        else:
            logging.error(f"{module_name.capitalize()} optimization error: {e}")
        
        logging.warning(f"Returning unoptimized {module_name} due to errors")
        return module

def optimize_analyzer(train_examples, output_dir=".", num_trials=5, num_bootstrapped_demos=3, num_candidates=5):
    return run_optimization(
        module=LinkedInStyleAnalyzer(),
        trainset=train_examples,
        metric=linkedin_style_metric,
        output_dir=output_dir,
        filename="optimized_linkedin_analyzer.json",
        module_name="analyzer",
        num_trials=num_trials,
        num_bootstrapped_demos=num_bootstrapped_demos,
        num_candidates=num_candidates
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
                        logging.info(f"Extracted prompt from {module_type.__name__} ({len(prompt)} chars)")
                        return prompt
                        
        logging.warning(f"Could not find prompt template in {module_type.__name__}")
    except Exception as e:
        logging.warning(f"Error extracting prompt from module: {e}")
        
    logging.info(f"Using default prompt for {module_type.__name__}")
    return default_prompt

def optimize_transformer(train_examples, analyzer, output_dir=".", num_trials=5, num_bootstrapped_demos=3, num_candidates=5):
    return run_optimization(
        module=LinkedInContentTransformer(),
        trainset=train_examples,
        metric=linkedin_content_metric,
        output_dir=output_dir,
        filename="optimized_linkedin_transformer.json",
        module_name="transformer",
        num_trials=num_trials,
        num_bootstrapped_demos=num_bootstrapped_demos,
        num_candidates=num_candidates
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