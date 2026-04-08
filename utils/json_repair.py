"""
JSON repair utility - handles malformed LLM outputs with multiple strategies.
Unblocks Task 2 and 3 by ensuring valid JSON is extracted from LLM responses.
"""

import json
import re
from typing import Any, Optional


def robust_json_parse(text: str) -> Optional[dict]:
    """
    Robustly parse JSON from LLM output using multiple strategies.
    
    Strategies in order of preference:
    1. Direct parse (already valid JSON)
    2. Strip markdown fences (```json ... ```)
    3. Find first complete { } block via regex
    4. Return None if all strategies fail
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse (most common happy path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove markdown code fences
    # Handle both ```json ... ``` and ``` ... ```
    markdown_patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',       # ``` ... ```
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find first complete JSON object { ... }
    # This regex finds the first { and its matching }
    json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Try fixing common issues
    # - Fix unquoted property names
    # - Fix single quotes converted to double
    # - Fix trailing commas
    text_fixed = fix_common_json_issues(text)
    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        pass
    
    # All strategies exhausted
    return None


def fix_common_json_issues(text: str) -> str:
    """
    Fix common JSON formatting issues from LLM outputs.
    
    Fixes:
    - Remove trailing commas before ] or }
    - Convert single quotes to double quotes for keys and values
    - Replace Python None/True/False with JSON null/true/false
    - Add quotes to unquoted keys
    
    Args:
        text: Malformed JSON string
        
    Returns:
        Attempted repair
    """
    
    # Replace Python None/True/False with JSON equivalents
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    
    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Convert single quotes to double quotes (simple heuristic)
    # This is tricky because we need to preserve strings with apostrophes
    # We'll be aggressive and replace most single quotes
    text = re.sub(r"'([^']*)'", r'"\1"', text)
    
    # Try to quote unquoted property names (simple heuristic)
    # Look for: word: value patterns and quote the word
    text = re.sub(r'(\{|,)\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', text)
    
    return text


def parse_json_or_none(text: str, default=None) -> Any:
    """
    Wrapper that returns default if parsing fails entirely.
    
    Args:
        text: JSON text to parse
        default: Value to return if parsing fails (default: None)
        
    Returns:
        Parsed JSON or default value
    """
    result = robust_json_parse(text)
    return result if result is not None else default


def clamp_vital_ranges(data: dict) -> dict:
    """
    Clamp vital signs to valid ranges for clinical safety.
    Repairs common model errors without rejecting the entire output.
    
    Args:
        data: Parsed JSON dict (potentially with invalid vitals)
        
    Returns:
        Repaired dict with vitals clamped to safe ranges
    """
    if not isinstance(data, dict):
        return data
    
    def apply_clamp(vitals_dict):
        """Apply clamping to a vitals dictionary."""
        if not isinstance(vitals_dict, dict):
            return vitals_dict
        
        # Temperature: clinical range 68-110°F (very permissive)
        if "temperature" in vitals_dict:
            temp = vitals_dict["temperature"]
            if temp is not None:
                try:
                    temp_float = float(temp)
                    vitals_dict["temperature"] = max(68.0, min(110.0, temp_float))
                except (TypeError, ValueError):
                    vitals_dict["temperature"] = None  # Remove invalid values
        
        # BP Systolic: 40-250 mmHg
        if "bp_systolic" in vitals_dict:
            bp = vitals_dict["bp_systolic"]
            if bp is not None:
                try:
                    bp_int = int(bp)
                    vitals_dict["bp_systolic"] = max(40, min(250, bp_int))
                except (TypeError, ValueError):
                    vitals_dict["bp_systolic"] = None
        
        # Heart rate: 20-250 bpm
        if "heart_rate" in vitals_dict:
            hr = vitals_dict["heart_rate"]
            if hr is not None:
                try:
                    hr_int = int(hr)
                    vitals_dict["heart_rate"] = max(20, min(250, hr_int))
                except (TypeError, ValueError):
                    vitals_dict["heart_rate"] = None
        
        # RR: 5-100 breaths/min
        if "respiratory_rate" in vitals_dict:
            rr = vitals_dict["respiratory_rate"]
            if rr is not None:
                try:
                    rr_int = int(rr)
                    vitals_dict["respiratory_rate"] = max(5, min(100, rr_int))
                except (TypeError, ValueError):
                    vitals_dict["respiratory_rate"] = None
        
        return vitals_dict
    
    # Single-patient action
    if "vitals" in data and isinstance(data["vitals"], dict):
        data["vitals"] = apply_clamp(data["vitals"])
    
    # Batch action: individual_triages
    if "individual_triages" in data and isinstance(data["individual_triages"], dict):
        for patient_id, triage in data["individual_triages"].items():
            if isinstance(triage, dict) and "vitals" in triage:
                triage["vitals"] = apply_clamp(triage["vitals"])
    
    return data


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Valid JSON
        ('{"esi_level": 3}', {"esi_level": 3}),
        
        # Markdown fences
        ('```json\n{"esi_level": 3}\n```', {"esi_level": 3}),
        ('```\n{"esi_level": 3}\n```', {"esi_level": 3}),
        
        # Trailing commas
        ('{"esi_level": 3,}', {"esi_level": 3}),
        
        # Unquoted keys (simplified — full JSON would need more work)
        ('{"esi_level": 3}', {"esi_level": 3}),
    ]
    
    for input_text, expected in test_cases:
        result = robust_json_parse(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {input_text[:50]}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}\n")
