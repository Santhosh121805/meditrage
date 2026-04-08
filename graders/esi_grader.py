"""
ESI Level grading with partial credit scoring.
"""


def score_esi(predicted: int, ground_truth: int) -> float:
    """
    Score ESI level prediction with partial credit.
    
    Args:
        predicted: Predicted ESI level (1-5)
        ground_truth: Ground truth ESI level (1-5)
    
    Returns:
        Score from 0.0 to 1.0
    """
    diff = abs(predicted - ground_truth)
    
    # Scoring rubric: off-by-one more acceptable than larger differences
    score_map = {
        0: 1.0,    # Perfect match
        1: 0.6,    # Off by one (clinically adjacent)
        2: 0.2,    # Dangerous under-triage or significant over-triage
        3: 0.0,    # Severe miss
        4: 0.0,    # Catastrophic miss
    }
    
    return score_map.get(diff, 0.0)
