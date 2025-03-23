import unittest
import dspy
from dspy_optimizer.utils.metrics import linkedin_style_metric, linkedin_content_metric, linkedin_quality_metric

class TestMetrics(unittest.TestCase):
    def test_linkedin_style_metric(self):
        # Test missing attributes
        prediction = dspy.Example()
        self.assertEqual(linkedin_style_metric(prediction), 0.0)
        
        # Test with short characteristics
        prediction = dspy.Example(style_characteristics="Short")
        self.assertEqual(linkedin_style_metric(prediction), 0.25)
        
        # Test with medium characteristics
        prediction = dspy.Example(style_characteristics="This text has an engaging tone with some LinkedIn-specific elements.")
        score = linkedin_style_metric(prediction)
        self.assertGreater(score, 0.5)
        
        # Test with comprehensive characteristics including LinkedIn elements
        prediction = dspy.Example(style_characteristics={
            "tone": "conversational and professional",
            "structure": "attention-grabbing hook followed by value points",
            "formatting": "uses bullet points and emojis",
            "hooks_and_cta": "strong question hook and call to engage",
            "emoji_usage": "professional emojis like 🚀, 💡, ✅"
        })
        score = linkedin_style_metric(prediction)
        self.assertGreater(score, 0.7)
    
    def test_linkedin_content_metric(self):
        # Test missing attributes
        prediction = dspy.Example()
        self.assertEqual(linkedin_content_metric(dspy.Example(), prediction), 0.0)
        
        # Test with empty article
        example = dspy.Example(content_to_transform="Original content", expected_linkedin_article="Expected article")
        prediction = dspy.Example(linkedin_article="")
        self.assertEqual(linkedin_content_metric(example, prediction), 0.0)
        
        # Test with unchanged content
        prediction = dspy.Example(linkedin_article="Original content")
        self.assertEqual(linkedin_content_metric(example, prediction), 0.1)
        
        # Test with good LinkedIn article
        prediction = dspy.Example(linkedin_article="🚀 This is a properly styled LinkedIn article with emojis ✅\n\nIt has multiple paragraphs and includes hashtags.\n\n• It uses bullet points\n• For better readability\n• And engagement\n\nWhat do you think about this approach? #LinkedIn #Engagement")
        score = linkedin_content_metric(example, prediction)
        self.assertGreaterEqual(score, 0.4)
    
    def test_linkedin_quality_metric(self):
        # Test missing attributes
        prediction = dspy.Example()
        self.assertEqual(linkedin_quality_metric(prediction), 0.0)
        
        # Test with minimal feedback and low score
        prediction = dspy.Example(quality_score=0.3, feedback="Short")
        score = linkedin_quality_metric(prediction)
        self.assertLess(score, 0.3)
        
        # Test with good feedback and high score
        prediction = dspy.Example(
            quality_score=0.8,
            feedback="To improve engagement, consider adding more emojis and a stronger hook in the beginning."
        )
        score = linkedin_quality_metric(prediction)
        self.assertGreater(score, 0.7)

if __name__ == "__main__":
    unittest.main()