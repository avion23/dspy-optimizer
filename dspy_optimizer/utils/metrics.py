import re
import json
from collections import Counter

LINKEDIN_ELEMENTS = [
    'tone', 'structure', 'formatting', 'emoji', 'hooks', 
    'cta', 'hashtags', 'bullets', 'questions', 'engagement'
]

LINKEDIN_EMOJIS = ['🔎', '💡', '🚀', '✅', '🤔', '👉', '💪', '📊', '🔑', '💼', '📈', '🔄', '📱', '💭', '⚡', '🎯', '💰', '🧠', '⭐']

POWER_WORDS = {
    'exclusive', 'secret', 'shocking', 'amazing', 'revolutionary', 'incredible', 
    'essential', 'crucial', 'vital', 'massive', 'powerful', 'proven', 'guaranteed',
    'extraordinary', 'remarkable', 'devastating', 'urgent', 'limited', 'unique',
    'breakthrough', 'instantly', 'skyrocket', 'explode', 'transform', 'announcing',
    'warning', 'danger', 'critical', 'fear', 'success', 'failure', 'mistake'
}

STRUCTURE_INDICATORS = {
    'problem': ['challenge', 'problem', 'issue', 'struggle', 'difficult', 'pain', 'risk'],
    'solution': ['solution', 'benefit', 'advantage', 'opportunity', 'results', 'outcome', 'success'],
    'evidence': ['example', 'study', 'research', 'data', 'survey', 'report', 'case', 'proof', 'evidence']
}

COMMON_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'as'}

CTA_INDICATORS = ['comment', 'share', 'thoughts', 'agree', 'follow', 'connect', 'learn', 'contact']

def linkedin_style_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    
    if not hasattr(prediction, 'style_characteristics'):
        return 0.0
    
    characteristics = prediction.style_characteristics
    
    if not characteristics:
        return 0.25
    
    if isinstance(characteristics, str):
        try:
            characteristics = json.loads(characteristics)
            if not isinstance(characteristics, dict):
                return 0.3
        except json.JSONDecodeError:
            if len(characteristics) < 20:
                return 0.25
                
            base_score = 0.4
            matches = sum(element.lower() in characteristics.lower() for element in LINKEDIN_ELEMENTS)
            structure_score = min(0.3, matches * 0.05)
            detail_score = min(0.3, (len(characteristics) / 300) * 0.3)
            return min(1.0, base_score + structure_score + detail_score)
    
    if not isinstance(characteristics, dict):
        return 0.3
    
    score = 0.4
    
    expected_keys = ['tone', 'structure', 'formatting', 'hooks_and_cta', 'emoji_usage']
    present_keys = [k for k in expected_keys if k in characteristics and characteristics[k]]
    
    completeness = len(present_keys) / len(expected_keys)
    score += completeness * 0.3
    
    total_length = sum(len(str(characteristics.get(k, ''))) for k in present_keys)
    detail_score = min(0.3, (total_length / 300) * 0.3)
    score += detail_score
    
    return min(score, 1.0)

def extract_key_topics(text):
    words = [w.strip('.,?!:;()[]{}"') for w in text.lower().split()]
    potential_topics = [w for w in words if w not in COMMON_WORDS and len(w) > 3]
    return [word for word, _ in Counter(potential_topics).most_common(5)]

def has_emoji(text):
    return any(emoji in text for emoji in LINKEDIN_EMOJIS)

def has_power_words(text):
    words = [word.strip('.,?!:;()[]{}"') for word in text.lower().split()]
    return any(word in POWER_WORDS for word in words)

def has_hashtags(text):
    return bool(re.search(r'#\w+', text))

def has_bullets(text):
    return bool(re.search(r'(•|\*|\-|\d+\.)\s', text))

def has_questions(text):
    return bool(re.search(r'\?\s', text))

def has_cta(text):
    return any(cta in text.lower() for cta in CTA_INDICATORS)

def linkedin_content_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    
    if not hasattr(prediction, 'linkedin_article'):
        return 0.0
    
    article = prediction.linkedin_article
    
    original = getattr(example, 'content_to_transform', '') if example else ''
    expected = getattr(example, 'expected_linkedin_article', '') if example else ''
    
    if not article:
        return 0.0
    if article == original:
        return 0.1
    
    score = 0.25
    
    # Content structure (25%)
    paragraphs = [p for p in article.split('\n\n') if p.strip()]
    if len(paragraphs) >= 3:
        score += 0.1
    
    if has_bullets(article):
        score += 0.15
    
    # Hook quality (15%)
    if paragraphs:
        first_line = paragraphs[0].split('\n')[0] if paragraphs[0].split('\n') else ""
        hook_score = 0.0
        
        if has_questions(first_line):
            hook_score += 0.05
            
        if has_power_words(first_line):
            hook_score += 0.05
            
        if has_emoji(first_line):
            hook_score += 0.05
            
        score += min(0.15, hook_score)
    
    # Content elements (35%)
    emoji_count = sum(1 for char in article if char in LINKEDIN_EMOJIS)
    if 2 <= emoji_count <= 8:
        score += 0.15
    elif emoji_count > 0:
        score += 0.05
    
    if has_hashtags(article):
        score += 0.1
    
    if paragraphs and has_cta(paragraphs[-1]):
        score += 0.1
    
    # Length assessment (10%)
    article_words = len(article.split())
    if 150 <= article_words <= 600:
        score += 0.1
    elif article_words < 50 or article_words > 1000:
        score -= 0.1
    
    # Topic relevance (5%)
    if original:
        original_topics = extract_key_topics(original)
        article_topics = extract_key_topics(article)
        topic_overlap = len(set(original_topics) & set(article_topics)) / max(1, len(original_topics))
        score += min(0.05, topic_overlap * 0.05)
    
    return min(max(score, 0.1), 1.0)

def linkedin_quality_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    
    if not hasattr(prediction, 'quality_score') or not hasattr(prediction, 'feedback'):
        return 0.0
    
    score = prediction.quality_score
    feedback = prediction.feedback
    
    if isinstance(score, str):
        try:
            score = float(score.replace('%', '')) / 100 if '%' in score else float(score)
        except ValueError:
            score = 0.5
    
    score = max(0.0, min(1.0, score))
    
    feedback_score = 0.5
    if feedback and len(feedback) > 50:
        feedback_score = 0.8
        if "improve" in feedback.lower() or "consider" in feedback.lower() or "suggest" in feedback.lower():
            feedback_score = 1.0
    
    return 0.7 * score + 0.3 * feedback_score