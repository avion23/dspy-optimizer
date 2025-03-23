import os
import json
from pathlib import Path
from dotenv import load_dotenv
import dspy
from typing import Dict, List, Optional
from dspy_optimizer.core.modules import StyleExtractor, StyleApplicator

def style_quality_metric(example=None, prediction=None, trace=None):
    """Modified to handle both 2 and 3 argument calls"""
    # Handle different calling conventions
    if prediction is None and example is not None:
        prediction = example  # When called with just one argument
        
    if not hasattr(prediction, 'style_characteristics'):
        return 0.0
    
    characteristics = prediction.style_characteristics
    if not characteristics or len(characteristics) < 20:
        return 0.25
    
    score = 0.5
    style_elements = [
        'tone', 'vocabulary', 'formality', 'sentence', 'paragraph', 
        'structure', 'punctuation', 'grammar', 'voice', 'person'
    ]
    
    # Handle string characteristics
    if isinstance(characteristics, str):
        for element in style_elements:
            if element in characteristics.lower():
                score += 0.05
    else:
        # Handle dictionary characteristics
        for element in style_elements:
            if element in characteristics and characteristics[element]:
                score += 0.05
    
    return min(score, 1.0)

def style_application_metric(example=None, prediction=None, trace=None):
    """Modified to handle both 2 and 3 argument calls"""
    # Handle different calling conventions
    if prediction is None and example is not None:
        prediction = example  # When called with just one argument
        example = None
        
    if not hasattr(prediction, 'styled_content'):
        return 0.0
    
    styled_content = prediction.styled_content
    
    # Get content from the right field (either content or content_to_style)
    original = ''
    expected = ''
    
    if example is not None:
        original = example.content if hasattr(example, 'content') else getattr(example, 'content_to_style', '')
        expected = getattr(example, 'expected_styled_content', '')
    
    if not styled_content:
        return 0.0
    if styled_content == original:
        return 0.1
    
    score = 0.4
    
    original_to_expected_ratio = len(expected) / len(original) if len(original) > 0 else 1
    styled_to_original_ratio = len(styled_content) / len(original) if len(original) > 0 else 1
    
    ratio_diff = abs(original_to_expected_ratio - styled_to_original_ratio)
    if ratio_diff < 0.2:
        score += 0.2
    
    original_words = set(original.lower().split())
    styled_words = set(styled_content.lower().split())
    
    intersection = len(original_words.intersection(styled_words))
    union = len(original_words.union(styled_words))
    
    if union > 0:
        iou = intersection / union
        if 0.2 <= iou <= 0.7:
            score += 0.3
    
    return min(score, 1.0)

def optimize_extractor(train_examples: List[dspy.Example], output_dir: str = ".") -> StyleExtractor:
    extractor = StyleExtractor()
    
    if len(train_examples) < 2:
        print("Not enough examples for optimization, returning unoptimized extractor")
        return extractor
        
    optimizer = dspy.MIPROv2(
        metric=style_quality_metric,
        max_bootstrapped_demos=4,
        auto="medium"
    )
    
    try:
        optimized_extractor = optimizer.compile(extractor, trainset=train_examples)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / "optimized_style_extractor.json"
        optimized_extractor.save(str(save_path))
        
        return optimized_extractor
    except Exception as e:
        print(f"Extractor optimization error: {e}")
        return extractor

def optimize_applicator(train_examples: List[dspy.Example], optimized_extractor: Optional[StyleExtractor] = None, 
                       output_dir: str = ".") -> StyleApplicator:
    extractor = optimized_extractor if optimized_extractor else StyleExtractor()
    applicator = StyleApplicator()
    
    application_examples = []
    for example in train_examples:
        try:
            extraction_result = extractor(example.sample_text)
            
            app_example = dspy.Example(
                content=example.content_to_style,
                style_characteristics=extraction_result.style_characteristics,
                expected_styled_content=example.expected_styled_content
            ).with_inputs("content", "style_characteristics")
            
            application_examples.append(app_example)
        except Exception as e:
            print(f"Example processing error: {e}")
            continue
    
    if not application_examples or len(application_examples) < 2:
        print("Not enough processed examples for applicator optimization, returning unoptimized applicator")
        return applicator
        
    optimizer = dspy.MIPROv2(
        metric=style_application_metric,
        max_bootstrapped_demos=4,
        auto="medium"
    )
    
    try:
        optimized_applicator = optimizer.compile(applicator, trainset=application_examples)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / "optimized_style_applicator.json"
        optimized_applicator.save(str(save_path))
        
        return optimized_applicator
    except Exception as e:
        print(f"Applicator optimization error: {e}")
        return applicator

def extract_prompt_from_module(module) -> str:
    # For unoptimized modules, return more useful default prompts
    if isinstance(module, StyleExtractor):
        return "You are an AI that analyzes text and creates concise style instructions for other AI's to rewrite texts. Extract key style characteristics such as tone, vocabulary level, sentence structure, and paragraph organization."
    elif isinstance(module, StyleApplicator):
        return "You are a helpful assistant for writing texts in specific styles. Apply the provided style characteristics to transform the content while preserving its core meaning."
        
    default_prompt = f"{module.__class__.__name__} optimized prompt"
    
    try:
        if hasattr(module, "_compiled_lm") and hasattr(module._compiled_lm, "program_template"):
            return module._compiled_lm.program_template
        
        if hasattr(module, "extractor") and hasattr(module.extractor, "template"):
            return module.extractor.template
        
        if hasattr(module, "applicator") and hasattr(module.applicator, "template"):
            return module.applicator.template
            
        if hasattr(module, "predictor") and hasattr(module.predictor, "template"):
            return module.predictor.template
            
        if hasattr(module, "lm") and hasattr(module.lm, "template"):
            return module.lm.template
    except Exception:
        pass
    
    return default_prompt

def extract_optimized_prompts(output_dir: str = ".") -> Dict[str, str]:
    output_path = Path(output_dir)
    extractor_path = output_path / "optimized_style_extractor.json"
    applicator_path = output_path / "optimized_style_applicator.json"
    
    extractor = StyleExtractor()
    applicator = StyleApplicator()
    
    if extractor_path.exists():
        try:
            extractor.load(str(extractor_path))
        except Exception:
            pass
    
    if applicator_path.exists():
        try:
            applicator.load(str(applicator_path))
        except Exception:
            pass
    
    prompts = {
        "style_analyzer_prompt": extract_prompt_from_module(extractor),
        "style_applicator_prompt": extract_prompt_from_module(applicator)
    }
    
    prompts_path = output_path / "optimized_prompts.json"
    try:
        with open(prompts_path, 'w') as f:
            json.dump(prompts, f, indent=2)
    except Exception:
        pass
    
    return prompts

def configure_lm():
    load_dotenv()
    
    if os.getenv("GEMINI_API_KEY"):
        try:
            return dspy.LM(
                model="openai/gemini-1.5-flash",
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv("GEMINI_API_KEY")
            )
        except Exception:
            pass
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            return dspy.LM(model="openai/gpt-4o-mini")
        except Exception:
            try:
                return dspy.LM(model="openai/gpt-3.5-turbo")
            except Exception:
                pass
    
    return dspy.LM()