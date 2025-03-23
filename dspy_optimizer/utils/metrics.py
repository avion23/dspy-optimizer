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
SPECIFIC_QUESTIONS = ['what do you think about', 'what has your experience been', 'share your']


def check_element_presence(text, elements):
    return sum(element in text.lower() for element in elements)


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
            matches = check_element_presence(characteristics, LINKEDIN_ELEMENTS)
            structure_score = min(0.3, matches * 0.1)
            detail_score = min(0.3, (len(characteristics) / 200) * 0.3)
            return min(1.0, base_score + structure_score + detail_score)
    
    if not isinstance(characteristics, dict):
        return 0.3
    
    score = 0.4
    
    expected_keys = ['tone', 'structure', 'formatting', 'hooks_and_cta', 'emoji_usage']
    present_keys = [k for k in expected_keys if k in characteristics and characteristics[k]]
    
    completeness = len(present_keys) / len(expected_keys)
    score += completeness * 0.3
    
    total_length = sum(len(str(characteristics.get(k, ''))) for k in present_keys)
    detail_score = min(0.3, (total_length / 200) * 0.3)
    score += detail_score
    
    return min(score, 1.0)


def extract_key_topics(text):
    words = [w.strip('.,?!:;()[]{}"') for w in text.lower().split()]
    potential_topics = [w for w in words if w not in COMMON_WORDS and len(w) > 3]
    return [word for word, _ in Counter(potential_topics).most_common(5)]


def has_power_words(text):
    words = [word.strip('.,?!:;()[]{}"') for word in text.lower().split()]
    return any(word in POWER_WORDS for word in words)


def has_statistic(text):
    number_pattern = r'\d+%|\d+|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion'
    return bool(re.search(number_pattern, text.lower()))


def emoji_distribution_score(text, emoji_list):
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return 0.0
    
    para_with_emoji = sum(1 for p in paragraphs if any(emoji in p for emoji in emoji_list))
    return min(1.0, para_with_emoji / len(paragraphs))


def analyze_sentence_length_variety(text):
    sentences = [s for s in re.split(r'[.!?]\s', text) if s.strip()]
    
    if len(sentences) <= 1:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    short = sum(1 for l in lengths if l <= 10)
    medium = sum(1 for l in lengths if 10 < l <= 20)
    
    if short > 0 and medium > 0:
        return 1.0
    elif short > 0 or medium > 0:
        return 0.5
    return 0.0


def calculate_hook_score(first_line, first_paragraph):
    hook_score = 0.0
    if '?' in first_line:
        hook_score += 0.4
    if has_statistic(first_line):
        hook_score += 0.4
    if has_power_words(first_line):
        hook_score += 0.3
    if any(emoji in first_line for emoji in LINKEDIN_EMOJIS):
        hook_score += 0.3
        
    if re.search(r'^(just|breaking|new|introducing|announcing)', first_line.lower()):
        hook_score += 0.3
    
    if re.search(r'(struggle|challenge|problem|difficulty|tired of)', first_paragraph.lower()):
        hook_score += 0.2
    
    return min(0.2, hook_score * 0.2)


def calculate_structure_score(paragraphs, article):
    structure_score = 0.0
    
    if len(paragraphs) >= 3:
        structure_score += 0.05
        
    if '•' in article or re.search(r'\n-\s', article) or re.search(r'\n\d\.\s', article):
        structure_score += 0.1
    
    first_third = ' '.join(paragraphs[:max(1, len(paragraphs)//3)]).lower()
    if any(p in first_third for p in STRUCTURE_INDICATORS['problem']):
        structure_score += 0.1
        
    middle_third = ' '.join(paragraphs[max(1, len(paragraphs)//3):max(2, 2*len(paragraphs)//3)]).lower()
    if any(s in middle_third for s in STRUCTURE_INDICATORS['solution']):
        structure_score += 0.1
        
    if any(e in article.lower() for e in STRUCTURE_INDICATORS['evidence']):
        structure_score += 0.1
    
    if re.search(r'[A-Z]{3,}', article) or re.search(r'\*\*.*?\*\*', article):
        structure_score += 0.05
        
    return min(0.3, structure_score)


def calculate_cta_score(last_paragraph):
    cta_score = 0.0
    
    if '?' in last_paragraph:
        cta_score += 0.4
        
    if any(cta in last_paragraph.lower() for cta in CTA_INDICATORS):
        cta_score += 0.4
        if any(specific in last_paragraph.lower() for specific in SPECIFIC_QUESTIONS):
            cta_score += 0.2
    
    if re.search(r'(comment|share|like|follow|connect|let me know|what do you think)', last_paragraph.lower()):
        cta_score += 0.3
            
    return min(0.2, cta_score * 0.2)


def calculate_hashtag_score(article, topics):
    hashtags = [word for word in article.split() if word.startswith('#')]
    
    if not hashtags:
        return 0.0
        
    hashtag_score = 0.0
    
    if 2 <= len(hashtags) <= 5:
        hashtag_score += 0.3
    elif len(hashtags) == 1:
        hashtag_score += 0.1
    elif len(hashtags) > 5:
        hashtag_score += 0.1
    
    last_paragraph = article.split('\n\n')[-1] if '\n\n' in article else article
    hashtags_at_end = all(hashtag in last_paragraph for hashtag in hashtags)
    if hashtags_at_end:
        hashtag_score += 0.2
        
    if topics:
        hashtag_text = ' '.join(h.lower()[1:] for h in hashtags)
        topic_matches = sum(1 for topic in topics if topic.lower() in hashtag_text)
        relevance = min(1.0, topic_matches / len(topics))
        hashtag_score += relevance * 0.3
    
    camel_case = sum(1 for h in hashtags if re.match(r'#[A-Z][a-z]+[A-Z][a-z]+', h))
    if camel_case / max(1, len(hashtags)) > 0.5:
        hashtag_score += 0.2
        
    return min(0.15, hashtag_score * 0.15)


def linkedin_content_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    example = None if prediction == example else example
    
    if not hasattr(prediction, 'linkedin_article'):
        return 0.0
    
    article = prediction.linkedin_article
    
    original = getattr(example, 'content_to_transform', '') if example else ''
    expected = getattr(example, 'expected_linkedin_article', '') if example else ''
    
    if not article:
        return 0.0
    if article == original:
        return 0.1
    
    score = 0.2
    topics = extract_key_topics(original or article)
    
    paragraphs = [p for p in article.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [article]
    
    lines = article.split('\n')
    first_line = lines[0] if lines else ""
    first_paragraph = paragraphs[0] if paragraphs else ""
    last_paragraph = paragraphs[-1] if paragraphs else ""
    
    score += calculate_hook_score(first_line, first_paragraph)
    score += calculate_structure_score(paragraphs, article)
    
    emoji_count = sum(1 for char in article if char in LINKEDIN_EMOJIS)
    emoji_distribution = emoji_distribution_score(article, LINKEDIN_EMOJIS)
    
    if 2 <= emoji_count <= 8:
        score += 0.1 + (emoji_distribution * 0.05)
    elif emoji_count > 0:
        score += 0.05
    
    score += calculate_hashtag_score(article, topics)
    score += calculate_cta_score(last_paragraph)
    
    article_words = len(article.split())
    if 150 <= article_words <= 600:
        score += 0.05
    
    if article_words < 50 or article_words > 1000:
        score -= 0.1
    
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
    
    feedback = prediction.feedback
    
    feedback_quality = 0.0
    if feedback and len(feedback) > 50:
        feedback_quality = 0.5
        quality_terms = ['improve', 'enhance', 'better', 'consider', 'suggest', 'engagement', 'hook', 'emoji']
        if any(term in feedback.lower() for term in quality_terms):
            feedback_quality = 0.8
            
        if re.search(r'(try adding|consider using|would benefit from)', feedback.lower()):
            feedback_quality = 1.0
    
    score = prediction.quality_score
    if isinstance(score, str):
        try:
            score = float(score.replace('%', '')) / 100 if '%' in score else float(score)
        except ValueError:
            score = 0.5
    
    score = max(0.0, min(1.0, score))
    
    return 0.7 * score + 0.3 * feedback_quality