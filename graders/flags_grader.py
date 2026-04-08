"""
Red flags grading with weighted recall.
"""

from typing import List


def score_flags(predicted: List[str], ground_truth: List[str]) -> float:
    """
    Score red flags prediction with emphasis on recall over precision.
    
    Missing a critical flag is worse than a false alarm.
    Scoring: (2 * recall + precision) / 3
    
    Args:
        predicted: Predicted red flags (list of coded strings)
        ground_truth: Ground truth red flags
    
    Returns:
        Score from 0.0 to 1.0
    """
    pred_set = set(predicted) if predicted else set()
    truth_set = set(ground_truth) if ground_truth else set()
    
    if not truth_set:
        # No flags to detect
        if not pred_set:
            return 1.0  # Correctly identified no flags
        else:
            return 0.0  # False alarms when none expected
    
    intersection = pred_set & truth_set
    
    # Recall: proportion of true flags that were detected
    recall = len(intersection) / len(truth_set)
    
    # Precision: proportion of predicted flags that were correct
    precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
    
    # Weighted average: recall weighted 2x over precision
    f1_weighted = (2 * recall + precision) / 3
    
    return round(f1_weighted, 4)
