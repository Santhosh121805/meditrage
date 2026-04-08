"""
Vitals extraction grading using F1 score.
"""

from models.schemas import Vitals


def score_vitals(predicted: Vitals, ground_truth: Vitals) -> float:
    """
    Score vitals extraction using F1 metric.
    
    A vital is considered correct if it matches within ±10% of ground truth.
    Missing fields count as false negatives.
    Hallucinated fields count as false positives.
    
    Args:
        predicted: Predicted vitals
        ground_truth: Ground truth vitals
    
    Returns:
        F1 score from 0.0 to 1.0
    """
    vital_fields = [
        "bp_systolic", "bp_diastolic", "heart_rate", "spo2",
        "temperature", "respiratory_rate", "gcs"
    ]
    
    tp = 0  # True positives: predicted and correct
    fn = 0  # False negatives: should have been predicted but weren't
    fp = 0  # False positives: predicted but shouldn't have been
    
    for field in vital_fields:
        pred_val = getattr(predicted, field)
        truth_val = getattr(ground_truth, field)
        
        if truth_val is not None:
            if pred_val is not None:
                # Check if within ±10%
                threshold = 0.1 * truth_val
                if abs(pred_val - truth_val) <= threshold:
                    tp += 1
                else:
                    # Wrong value - counts as both FN and FP
                    fn += 1
                    fp += 1
            else:
                # Missing value
                fn += 1
        else:
            if pred_val is not None:
                # Hallucinated field
                fp += 1
    
    # Calculate F1 score
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)
