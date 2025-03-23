import re
from collections import Counter

def linkedin_style_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    
    if not hasattr(prediction, 'style_characteristics'):
        return 0.0
    
    characteristics = prediction.style_characteristics
    if not characteristics or (isinstance(characteristics, str) and len(characteristics) < 20):
        return 0.25
    
    score = 0.5
    linkedin_elements = [
        'tone', 'structure', 'formatting', 'emoji', 'hooks', 
        'cta', 'hashtags', 'bullets', 'questions', 'engagement'
    ]
    
    if isinstance(characteristics, str):
        matches = sum(element in characteristics.lower() for element in linkedin_elements)
        score += min(0.5, matches * 0.05)
    else:
        matches = sum(any(element in key.lower() for key in characteristics.keys() if characteristics.get(key)) 
                      for element in linkedin_elements)
        score += min(0.5, matches * 0.05)
    
    return min(score, 1.0)

def extract_key_topics(text):
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'as'}
    words = [w.strip('.,?!:;()[]{}"') for w in text.lower().split()]
    potential_topics = [w for w in words if w not in common_words and len(w) > 3]
    return [word for word, _ in Counter(potential_topics).most_common(5)]

def has_power_words(text):
    power_words = {
        'exclusive', 'secret', 'shocking', 'amazing', 'revolutionary', 'incredible', 
        'essential', 'crucial', 'vital', 'massive', 'powerful', 'proven', 'guaranteed',
        'extraordinary', 'remarkable', 'devastating', 'urgent', 'limited', 'unique',
        'breakthrough', 'instantly', 'skyrocket', 'explode', 'transform', 'announcing',
        'warning', 'danger', 'critical', 'fear', 'success', 'failure', 'mistake'
    }
    
    words = [word.strip('.,?!:;()[]{}"') for word in text.lower().split()]
    return any(word in power_words for word in words)

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
    linkedin_emojis = ['🔎', '💡', '🚀', '✅', '🤔', '👉', '💪', '📊', '🔑', '💼', '📈', '🔄', '📱', '💭', '⚡', '🎯', '💰', '🧠', '⭐']
    topics = extract_key_topics(original)
    
    lines = article.split('\n')
    paragraphs = [p for p in article.split('\n\n') if p.strip()]
    first_paragraph = paragraphs[0] if paragraphs else ""
    last_paragraph = paragraphs[-1] if paragraphs else ""
    
    first_line = lines[0] if lines else ""
    hook_score = 0.0
    if '?' in first_line:
        hook_score += 0.5
    if has_statistic(first_line):
        hook_score += 0.5
    if has_power_words(first_line):
        hook_score += 0.3
    if any(emoji in first_line for emoji in linkedin_emojis):
        hook_score += 0.2
    score += min(0.15, hook_score)
    
    emoji_count = sum(1 for char in article if char in linkedin_emojis)
    emoji_distribution = emoji_distribution_score(article, linkedin_emojis)
    
    if 3 <= emoji_count <= 7:
        score += 0.1 + (emoji_distribution * 0.05)
    elif emoji_count > 0:
        score += 0.05
    
    hashtags = [word for word in article.split() if word.startswith('#')]
    if 2 <= len(hashtags) <= 5:
        hashtag_score = 0.05
        
        if topics:
            hashtag_text = ' '.join(h.lower()[1:] for h in hashtags)
            topic_matches = sum(1 for topic in topics if topic.lower() in hashtag_text)
            hashtag_relevance = min(1.0, topic_matches / len(topics))
            hashtag_score += hashtag_relevance * 0.1
        
        score += hashtag_score
    
    questions = sum(1 for line in lines if '?' in line)
    if questions >= 1:
        score += 0.1 if '?' in last_paragraph else 0.05
    
    indicators = {
        'problem': ['challenge', 'problem', 'issue', 'struggle', 'difficult', 'pain', 'risk'],
        'solution': ['solution', 'benefit', 'advantage', 'opportunity', 'results', 'outcome', 'success'],
        'evidence': ['example', 'study', 'research', 'data', 'survey', 'report', 'case', 'proof', 'evidence']
    }
    
    structure_score = 0.0
    if any(p in ' '.join(lines[:min(5, len(lines))]).lower() for p in indicators['problem']):
        structure_score += 0.3
    if any(s in ' '.join(lines[min(3, len(lines)-1):min(10, len(lines))]).lower() for s in indicators['solution']):
        structure_score += 0.3
    if any(e in article.lower() for e in indicators['evidence']):
        structure_score += 0.2
        
    score += min(0.15, structure_score)
    
    bullet_lines = sum(1 for line in lines if line.strip().startswith(('•', '-', '✅', '✓', '→', '1.', '2.', '3.', '4.', '5.')))
    score += 0.1 if bullet_lines >= 3 else (0.05 if bullet_lines > 0 else 0)
    
    if paragraphs:
        max_para_len = max(len(p) for p in paragraphs)
        score += 0.1 if max_para_len < 150 else (0.05 if max_para_len < 250 else 0)
    
    score += analyze_sentence_length_variety(article) * 0.05
    
    cta_score = 0.0
    if '?' in last_paragraph:
        cta_score += 0.5
    if any(cta in last_paragraph.lower() for cta in ['comment', 'share', 'thoughts', 'agree', 'follow', 'connect', 'learn', 'contact']):
        cta_score += 0.5
        if any(specific in last_paragraph.lower() for specific in ['what do you think about', 'what has your experience been', 'share your']):
            cta_score += 0.2
    score += min(0.1, cta_score / 10)
    
    newline_pairs = article.count('\n\n')
    score += 0.1 if newline_pairs >= 4 else (0.05 if newline_pairs >= 2 else 0)
    
    if expected:
        if len(article) < len(expected) * 0.5:
            score -= 0.2
        elif len(article) > len(expected) * 1.5:
            score -= 0.1
    
    return min(max(score, 0.0), 1.0)

def linkedin_quality_metric(example=None, prediction=None, trace=None):
    prediction = example if prediction is None and example else prediction
    
    if not hasattr(prediction, 'quality_score') or not hasattr(prediction, 'feedback'):
        return 0.0
    
    feedback = prediction.feedback
    feedback_quality = 0.0
    
    if feedback and len(feedback) > 50:
        feedback_quality = 0.5
        quality_terms = ['improve', 'enhance', 'better', 'consider', 'suggest', 'engagement', 'hook', 'emoji']
        if any(term in feedback for term in quality_terms):
            feedback_quality = 0.8
    
    score = prediction.quality_score
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            score = 0.5
    
    return 0.7 * score + 0.3 * feedback_quality