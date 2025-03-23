import dspy
import json

class LinkedInStyleAnalysis(dspy.Signature):
    sample_post = dspy.InputField()
    style_characteristics = dspy.OutputField()

class LinkedInContentTransformation(dspy.Signature):
    content_to_transform = dspy.InputField()
    style_characteristics = dspy.InputField()
    linkedin_article = dspy.OutputField()

class ArticleQualityEvaluation(dspy.Signature):
    original_sample = dspy.InputField()
    generated_article = dspy.InputField()
    quality_score = dspy.OutputField()
    feedback = dspy.OutputField()

class StyleExtraction(dspy.Signature):
    sample_text = dspy.InputField()
    style_characteristics = dspy.OutputField()

class StyleApplication(dspy.Signature):
    content = dspy.InputField()
    style_characteristics = dspy.InputField()
    styled_content = dspy.OutputField()

class StyleSimilarityEvaluation(dspy.Signature):
    original_style = dspy.InputField()
    styled_content = dspy.InputField()
    similarity_score = dspy.OutputField()
    feedback = dspy.OutputField()

class LinkedInStyleAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(LinkedInStyleAnalysis)
    
    def forward(self, sample_post):
        result = self.analyzer(sample_post=sample_post)
        
        if isinstance(result.style_characteristics, str):
            try:
                result.style_characteristics = json.loads(result.style_characteristics)
            except json.JSONDecodeError:
                result.style_characteristics = {
                    "tone": "extracted from sample",
                    "structure": "extracted from sample",
                    "formatting": "extracted from sample",
                    "hooks_and_cta": "extracted from sample",
                    "emoji_usage": "extracted from sample"
                }
        
        return result

class LinkedInContentTransformer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.transformer = dspy.Predict(LinkedInContentTransformation)
    
    def forward(self, content_to_transform, style_characteristics):
        return self.transformer(
            content_to_transform=content_to_transform, 
            style_characteristics=style_characteristics
        )

class ArticleQualityEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(ArticleQualityEvaluation)
    
    def forward(self, original_sample, generated_article):
        result = self.evaluator(
            original_sample=original_sample, 
            generated_article=generated_article
        )
        
        if isinstance(result.quality_score, str):
            try:
                result.quality_score = float(result.quality_score)
            except ValueError:
                result.quality_score = 0.5
        
        return result

class LinkedInArticlePipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = LinkedInStyleAnalyzer()
        self.transformer = LinkedInContentTransformer()
        self.evaluator = ArticleQualityEvaluator()
    
    def forward(self, sample_post, content_to_transform):
        analysis_result = self.analyzer(sample_post)
        transformation_result = self.transformer(
            content_to_transform=content_to_transform, 
            style_characteristics=analysis_result.style_characteristics
        )
        evaluation_result = self.evaluator(
            original_sample=sample_post,
            generated_article=transformation_result.linkedin_article
        )
        
        return {
            "style_characteristics": analysis_result.style_characteristics,
            "linkedin_article": transformation_result.linkedin_article,
            "quality_score": evaluation_result.quality_score,
            "feedback": evaluation_result.feedback
        }

class StyleExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(StyleExtraction)
    
    def forward(self, sample_text):
        result = self.extractor(sample_text=sample_text)
        
        if isinstance(result.style_characteristics, str):
            try:
                result.style_characteristics = json.loads(result.style_characteristics)
            except json.JSONDecodeError:
                result.style_characteristics = {
                    "tone": "extracted from sample",
                    "vocabulary": "extracted from sample",
                    "sentence_structure": "extracted from sample",
                    "formality": "extracted from sample"
                }
        
        return result

class StyleApplicator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.applicator = dspy.Predict(StyleApplication)
    
    def forward(self, content, style_characteristics):
        return self.applicator(
            content=content, 
            style_characteristics=style_characteristics
        )

class StyleEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(StyleSimilarityEvaluation)
    
    def forward(self, original_style, styled_content):
        result = self.evaluator(
            original_style=original_style, 
            styled_content=styled_content
        )
        
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
            original_style=sample_text,
            styled_content=application_result.styled_content
        )
        
        return {
            "style_characteristics": extraction_result.style_characteristics,
            "styled_content": application_result.styled_content,
            "similarity_score": evaluation_result.similarity_score,
            "feedback": evaluation_result.feedback
        }