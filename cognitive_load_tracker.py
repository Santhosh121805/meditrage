"""
Cognitive load tracker - detects clarification requests in agent reasoning.
Bonus feature: automatic clarification counting without API calls.
"""

import re


def detect_clarifications(reasoning: str) -> int:
    """
    Detect and count clarification request phrases in reasoning text.
    
    Phrases like "need more info", "please clarify", "cannot determine without"
    auto-increment clarification counter.
    
    Args:
        reasoning: The reasoning text from TriageAction.reasoning
    
    Returns:
        Number of clarification requests detected
    """
    
    clarification_phrases = [
        r"need\s+more\s+info",
        r"need\s+clarif",
        r"please\s+clarif",
        r"cannot\s+determine\s+without",
        r"cannot\s+decide\s+without",
        r"unclear",
        r"ambiguous",
        r"\?{2,}",  # Multiple question marks
        r"need\s+to\s+know",
        r"insufficient\s+information",
        r"require\s+clarif",
    ]
    
    count = 0
    reasoning_lower = reasoning.lower()
    
    for phrase in clarification_phrases:
        matches = re.findall(phrase, reasoning_lower)
        count += len(matches)
    
    return count
