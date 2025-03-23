import pytest
import dspy
import os
from dotenv import load_dotenv
from dspy_optimizer.core.modules import StyleExtractor, StyleApplicator, StylePipeline

def style_effectiveness_metric(prediction, trace=None):
    """Combined metric for evaluating style transfer pipeline."""
    if not hasattr(prediction, 'similarity_score') or not hasattr(prediction, 'feedback'):
        return 0.0
    
    feedback_quality = 0.0
    if hasattr(prediction, 'feedback'):
        feedback = prediction.feedback.lower()
        if len(feedback) > 50:
            feedback_quality = 0.5
            
            quality_terms = ['improve', 'enhance', 'better', 'consider', 'suggest']
            for term in quality_terms:
                if term in feedback:
                    feedback_quality = 0.8
                    break
    
    # Convert similarity score to float if it's a string
    score = prediction.similarity_score
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            score = 0.5
    
    return 0.7 * score + 0.3 * feedback_quality

@pytest.fixture(scope="module")
def configure_dspy():
    load_dotenv()
    if os.getenv("GEMINI_API_KEY"):
        lm = dspy.LM(
            model="openai/gemini-1.5-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        lm = dspy.LM(model="openai/gpt-4o-mini")
    
    dspy.settings.configure(lm=lm)
    return lm

@pytest.fixture
def style_samples():
    return [
        {
            "name": "formal_academic",
            "sample": "The results indicate a statistically significant correlation between variables X and Y (p < 0.01). Furthermore, the analysis reveals several underlying factors that warrant further investigation.",
            "content_to_style": "This shows X and Y are related. We found some interesting things worth looking into more."
        },
        {
            "name": "casual_conversational",
            "sample": "Hey there! Just wanted to let you know that I checked out that new restaurant we talked about. It was pretty awesome, though a bit pricey. The desserts were to die for!",
            "content_to_style": "I visited the restaurant we discussed previously. It was excellent but expensive. The desserts were exceptional."
        },
        {
            "name": "technical_documentation",
            "sample": "To implement this feature, invoke the `configure()` method with the appropriate parameters. Ensure that all dependencies are properly initialized before proceeding. See example 3.2 for reference implementation.",
            "content_to_style": "Use the configure function with the right settings. Make sure everything is set up first. Look at example 3.2."
        }
    ]

def test_style_extractor(configure_dspy, style_samples):
    extractor = StyleExtractor()
    
    for sample in style_samples:
        result = extractor(sample["sample"])
        
        assert isinstance(result.style_characteristics, dict)
        assert len(result.style_characteristics) > 0
        assert "tone" in result.style_characteristics
        assert "vocabulary" in result.style_characteristics
        
        print(f"Extracted style for {sample['name']}: {result.style_characteristics}")

def test_style_applicator(configure_dspy, style_samples):
    extractor = StyleExtractor()
    applicator = StyleApplicator()
    
    for sample in style_samples:
        extraction_result = extractor(sample["sample"])
        
        application_result = applicator(
            content=sample["content_to_style"],
            style_characteristics=extraction_result.style_characteristics
        )
        
        assert isinstance(application_result.styled_content, str)
        assert len(application_result.styled_content) > 0
        assert application_result.styled_content != sample["content_to_style"]
        
        print(f"Original: {sample['content_to_style']}")
        print(f"Styled: {application_result.styled_content}")
        print("---")

def test_style_similarity(configure_dspy, style_samples):
    pipeline = StylePipeline()
    
    for sample in style_samples:
        result = pipeline(
            sample_text=sample["sample"],
            content_to_style=sample["content_to_style"]
        )
        
        if isinstance(result["similarity_score"], str):
            try:
                result["similarity_score"] = float(result["similarity_score"])
            except ValueError:
                result["similarity_score"] = 0.5
        
        assert isinstance(result["similarity_score"], float)
        assert 0 <= result["similarity_score"] <= 1.0
        assert result["similarity_score"] > 0.5, f"Style transfer for {sample['name']} not effective"
        
        print(f"Similarity score for {sample['name']}: {result['similarity_score']}")
        print(f"Feedback: {result['feedback']}")

@pytest.mark.parametrize("component", ["extractor", "applicator"])
def test_optimize_component(configure_dspy, style_samples, component):
    train_examples = []
    for sample in style_samples[:2]:
        train_examples.append(
            dspy.Example(
                sample_text=sample["sample"],
                content_to_style=sample["content_to_style"],
                expected_similarity_score=0.9,
                expected_feedback="Excellent style matching"
            ).with_inputs("sample_text", "content_to_style")
        )
    
    test_examples = []
    for sample in style_samples[2:]:
        test_examples.append(
            dspy.Example(
                sample_text=sample["sample"],
                content_to_style=sample["content_to_style"],
                expected_similarity_score=0.9,
                expected_feedback="Excellent style matching"
            ).with_inputs("sample_text", "content_to_style")
        )
    
    if component == "extractor":
        module = StyleExtractor()
    else:
        module = StyleApplicator()
    
    assert hasattr(module, "save"), f"{component} module should support save"