"""
Queue ranking grading using Kendall's tau and resource constraints.
Includes graceful degradation for partial/invalid data.
"""

from typing import Dict, List, Tuple, Optional
from scipy.stats import kendalltau


def score_queue(
    predicted_order: List[str],
    ground_truth_order: List[str],
    resource_allocation: Dict[str, str],
    bed_counts: Dict[str, int],
) -> float:
    """
    Score queue ordering using Kendall's tau correlation.
    Apply penalty for resource constraint violations.
    Gracefully handles partial/invalid data.
    
    Args:
        predicted_order: Predicted priority queue (patient IDs, 1st = most urgent)
        ground_truth_order: Ground truth ordering
        resource_allocation: Dict mapping patient_id → assigned ward/bay
        bed_counts: Dict mapping ward → available beds
    
    Returns:
        Score from 0.0 to 1.0 after constraint penalty
    """
    
    # Graceful degradation: return partial score if data is invalid
    if not predicted_order or len(predicted_order) == 0:
        return 0.0  # No queue provided
    
    if not ground_truth_order or len(ground_truth_order) == 0:
        # No ground truth - can't score ranking, return 0.2 for partial credit
        return 0.2
    
    # Filter: only score patients that are in BOTH predicted and ground truth
    valid_patients = set(predicted_order) & set(ground_truth_order)
    if len(valid_patients) == 0:
        return 0.1  # No overlap - minimal credit
    
    # Partial ordering: use only valid patients
    predicted_subset = [p for p in predicted_order if p in valid_patients]
    truth_subset = ground_truth_order[:len(valid_patients)]  # Take first N from truth
    
    if len(predicted_subset) < 2:
        # Cannot compute correlation with < 2 items
        rank_score = 0.3
    else:
        try:
            # Compute Kendall's tau rank correlation
            truth_ranks = [truth_subset.index(p) if p in truth_subset else len(truth_subset) 
                          for p in predicted_subset]
            pred_ranks = list(range(len(predicted_subset)))
            
            tau, _ = kendalltau(truth_ranks, pred_ranks)
            # Normalize tau from [-1, 1] to [0, 1]
            rank_score = (tau + 1) / 2
        except (ValueError, IndexError):
            # Fallback if correlation computation fails
            rank_score = 0.2
    
    # Compute constraint violation penalty
    # -0.3 points for each patient assigned to a full ward (bed_count = 0)
    constraint_violations = 0
    if resource_allocation and bed_counts:
        for patient_id, ward in resource_allocation.items():
            available_beds = bed_counts.get(ward, 1)  # Default to 1 available if not specified
            if available_beds <= 0:
                constraint_violations += 1
    
    constraint_penalty = 0.3 * constraint_violations
    
    # Final score: minimum 0.15 for partial effort, maximum 1.0
    calculated_score = max(0.0, rank_score - constraint_penalty)
    final_score = max(0.15, calculated_score)  # Graceful minimum
    return round(final_score, 4)
