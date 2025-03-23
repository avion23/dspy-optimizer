import dspy
import json

class StyleExtraction(dspy.Signature):
    sample_text = dspy.InputField()
    style_characteristics = dspy.OutputField()

class StyleApplication(dspy.Signature):
    content = dspy.InputField()
    style_characteristics = dspy.InputField()
    styled_content = dspy.OutputField()

class StyleSimilarity(dspy.Signature):
    original_sample = dspy.InputField()
    styled_content = dspy.InputField()
    similarity_score = dspy.OutputField()
    feedback = dspy.OutputField()

class StyleExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(StyleExtraction)
    
    def forward(self, sample_text):
        result = self.extractor(sample_text=sample_text)
        
        # Convert string to dict if needed (for Gemini compatibility)
        if isinstance(result.style_characteristics, str):
            try:
                # Try to parse it as JSON first
                result.style_characteristics = json.loads(result.style_characteristics)
            except json.JSONDecodeError:
                # If not JSON, convert to a simple dict with features
                style_text = result.style_characteristics
                result.style_characteristics = {
                    "tone": style_text,
                    "vocabulary": "extracted from text",
                    "formality": "extracted from text",
                    "sentence": "extracted from text",
                    "paragraph": "extracted from text"
                }
        
        return result

class StyleApplicator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.applicator = dspy.Predict(StyleApplication)
    
    def forward(self, content, style_characteristics):
        return self.applicator(content=content, style_characteristics=style_characteristics)

class StyleEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(StyleSimilarity)
    
    def forward(self, original_sample, styled_content):
        result = self.evaluator(original_sample=original_sample, styled_content=styled_content)
        
        # Convert similarity score to float if it's a string
        if isinstance(result.similarity_score, str):
            try:
                result.similarity_score = float(result.similarity_score)
            except ValueError:
                result.similarity_score = 0.5
        
        return result

class StylePipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = StyleExtractor()
        self.applicator = StyleApplicator()
        self.evaluator = StyleEvaluator()
    
    def forward(self, sample_text, content_to_style):
        extraction_result = self.extractor(sample_text)
        application_result = self.applicator(
            content=content_to_style, 
            style_characteristics=extraction_result.style_characteristics
        )
        evaluation_result = self.evaluator(
            original_sample=sample_text,
            styled_content=application_result.styled_content
        )
        
        return {
            "style_characteristics": extraction_result.style_characteristics,
            "styled_content": application_result.styled_content,
            "similarity_score": evaluation_result.similarity_score,
            "feedback": evaluation_result.feedback
        }