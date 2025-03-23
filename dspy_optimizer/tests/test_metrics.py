import unittest
import json
from unittest.mock import patch
import dspy
from dspy_optimizer.utils.metrics import (
    linkedin_style_metric, 
    linkedin_content_metric, 
    linkedin_quality_metric
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.sample_post = """🚀 Excited to share that our team launched a new feature today!

This has been 6 months in the making, with countless iterations based on YOUR feedback.

What's new:
• Real-time collaboration
• AI-powered suggestions
• Cross-platform integration

The best part? It's available to all users starting TODAY!

What feature would you like to see next? Drop your ideas below! 👇 #ProductLaunch #Innovation"""

        self.original_content = """We've developed a new project management methodology that combines agile and waterfall approaches. It allows teams to plan long-term while maintaining flexibility for changing requirements. The methodology emphasizes clear documentation, regular team check-ins, and continuous improvement cycles. We've tested it with 5 enterprise clients so far with positive results."""

        self.transformed_content = """🚀 Just developed a GAME-CHANGING project management methodology!

Ever felt stuck between rigid waterfall planning and sometimes-chaotic agile? We've cracked the code by combining the best of both worlds!

What our hybrid approach delivers:
• Long-term planning WITH flexibility for changing requirements
• Clear documentation that actually helps your team
• Strategic check-ins that don't waste time
• Built-in continuous improvement cycles

The proof? We've already tested with 5 enterprise clients and the results speak for themselves!

Has your team struggled with finding the right project management balance? What approaches have worked for you? Share below! 👇

#ProjectManagement #AgileMethodology #ProductivityHacks #TeamEfficiency"""

        self.style_characteristics = {
            "tone": "Professional yet enthusiastic, using a conversational style",
            "structure": "Short, scannable paragraphs with a clear intro hook and conclusion CTA",
            "formatting": "Strategic use of line breaks, bullet points, and occasional capitals for emphasis",
            "hooks_and_cta": "Opens with attention-grabbing statement, closes with question to drive engagement",
            "emoji_usage": "Selective use of relevant emojis (🚀, 👇) to add visual interest"
        }

    def test_linkedin_style_metric(self):
        example = dspy.Example(
            sample_post=self.sample_post,
            style_characteristics=self.style_characteristics
        )
        
        score = linkedin_style_metric(prediction=example)
        self.assertGreater(score, 0.7, "Style metric should return a high score for good style characteristics")
        
        example_empty = dspy.Example(
            sample_post=self.sample_post,
            style_characteristics={}
        )
        score_empty = linkedin_style_metric(prediction=example_empty)
        self.assertLess(score_empty, 0.6, "Style metric should return a lower score for empty characteristics")
        
        example_str = dspy.Example(
            sample_post=self.sample_post,
            style_characteristics="Professional tone with emojis"
        )
        score_str = linkedin_style_metric(prediction=example_str)
        self.assertGreater(score_str, 0.5, "Style metric should handle string characteristics")
        self.assertLess(score_str, score, "String characteristics should score lower than detailed dict")

    def test_linkedin_content_metric(self):
        example = dspy.Example(
            content_to_transform=self.original_content,
            expected_linkedin_article=self.transformed_content,
            linkedin_article=self.transformed_content
        )
        
        score = linkedin_content_metric(prediction=example)
        self.assertGreater(score, 0.6, "Content metric should return a high score for good transformation")
        
        example_same = dspy.Example(
            content_to_transform=self.original_content,
            expected_linkedin_article=self.transformed_content,
            linkedin_article=self.original_content
        )
        score_same = linkedin_content_metric(prediction=example_same)
        self.assertLessEqual(score_same, 0.6, "Content metric should return a moderate/low score when minimal transformation occurred")
        
        example_empty = dspy.Example(
            content_to_transform=self.original_content,
            expected_linkedin_article=self.transformed_content,
            linkedin_article=""
        )
        score_empty = linkedin_content_metric(prediction=example_empty)
        self.assertEqual(score_empty, 0.0, "Content metric should return zero for empty transformation")
        
        no_emoji_content = self.transformed_content.replace("🚀", "").replace("👇", "")
        example_no_emoji = dspy.Example(
            content_to_transform=self.original_content,
            expected_linkedin_article=self.transformed_content,
            linkedin_article=no_emoji_content
        )
        score_no_emoji = linkedin_content_metric(prediction=example_no_emoji)
        self.assertLess(score_no_emoji, score, "Content without emojis should score lower")

    def test_linkedin_quality_metric(self):
        example = dspy.Example(
            quality_score=0.85,
            feedback="Good transformation that effectively uses LinkedIn best practices. The hook is strong and the emoji usage is appropriate. Consider adding more specific hashtags."
        )
        
        score = linkedin_quality_metric(prediction=example)
        self.assertGreater(score, 0.7, "Quality metric should return a high score for good quality score and feedback")
        
        example_low = dspy.Example(
            quality_score=0.3,
            feedback="Poor transformation with minimal LinkedIn best practices applied."
        )
        score_low = linkedin_quality_metric(prediction=example_low)
        self.assertLess(score_low, 0.5, "Quality metric should return a low score for low quality score")
        
        example_str_score = dspy.Example(
            quality_score="0.75",
            feedback="Good transformation overall."
        )
        score_str = linkedin_quality_metric(prediction=example_str_score)
        self.assertGreaterEqual(score_str, 0.5, "Quality metric should handle string quality scores")
        
        example_no_feedback = dspy.Example(
            quality_score=0.9,
            feedback=""
        )
        score_no_feedback = linkedin_quality_metric(prediction=example_no_feedback)
        self.assertGreater(score_no_feedback, 0.6, "Quality metric should handle empty feedback")
        self.assertLessEqual(score_no_feedback, score, "Quality without feedback should score lower or equal")

if __name__ == "__main__":
    unittest.main()