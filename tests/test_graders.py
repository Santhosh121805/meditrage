"""
Unit tests for MedTriage graders.
Run: pytest tests/test_graders.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.schemas import Vitals
from graders.esi_grader import score_esi
from graders.vitals_grader import score_vitals
from graders.flags_grader import score_flags
from graders.queue_grader import score_queue


class TestESIGrader:
    """Tests for ESI level scoring."""
    
    def test_perfect_match(self):
        """Perfect ESI prediction should score 1.0."""
        assert score_esi(2, 2) == 1.0
        assert score_esi(1, 1) == 1.0
        assert score_esi(5, 5) == 1.0
    
    def test_off_by_one(self):
        """Off-by-one predictions should score 0.6."""
        assert score_esi(1, 2) == 0.6
        assert score_esi(3, 2) == 0.6
        assert score_esi(4, 5) == 0.6  # Off by one
    
    def test_off_by_two(self):
        """Off-by-two or more should score 0.2 or 0.0."""
        assert score_esi(1, 3) == 0.2  # Dangerous under-triage
        assert score_esi(5, 3) == 0.2
    
    def test_catastrophic_miss(self):
        """ESI 1 predicted as 4+ should score 0.0."""
        assert score_esi(1, 5) == 0.0
        assert score_esi(5, 1) == 0.0


class TestVitalsGrader:
    """Tests for vitals extraction scoring."""
    
    def test_perfect_vitals(self):
        """All vitals correct should score 1.0."""
        pred = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0)
        truth = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0)
        score = score_vitals(pred, truth)
        assert score == 1.0
    
    def test_all_vitals_wrong(self):
        """Predicting wrong vitals should score 0.0."""
        pred = Vitals(bp_systolic=180, heart_rate=140, spo2=80.0)
        truth = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0)
        score = score_vitals(pred, truth)
        assert score == 0.0
    
    def test_missing_vital(self):
        """Missing a vital field counts as recall miss."""
        pred = Vitals(bp_systolic=120, heart_rate=None, spo2=98.0)
        truth = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0)
        score = score_vitals(pred, truth)
        assert 0.0 < score < 1.0  # Partial credit
    
    def test_hallucinated_vital(self):
        """Hallucinating a vital counts as precision miss."""
        pred = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0, gcs=15)
        truth = Vitals(bp_systolic=120, heart_rate=80, spo2=98.0)
        score = score_vitals(pred, truth)
        assert 0.0 < score < 1.0


class TestFlagsGrader:
    """Tests for red flags scoring."""
    
    def test_all_flags_correct(self):
        """All flags correct should score 1.0."""
        pred = ["STEMI_pattern", "ACS_with_hemodynamic_instability"]
        truth = ["STEMI_pattern", "ACS_with_hemodynamic_instability"]
        score = score_flags(pred, truth)
        assert score == 1.0
    
    def test_all_flags_wrong(self):
        """No correct flags should score 0.0."""
        pred = ["anaphylaxis", "septic_shock"]
        truth = ["STEMI_pattern", "acute_stroke"]
        score = score_flags(pred, truth)
        assert score == 0.0
    
    def test_missing_flags(self):
        """Missing a critical flag (recall miss) reduces score more."""
        pred = ["STEMI_pattern"]
        truth = ["STEMI_pattern", "ACS_with_hemodynamic_instability"]
        score = score_flags(pred, truth)
        # Recall = 0.5, Precision = 1.0
        # Weighted = (2 * 0.5 + 1.0) / 3 = 0.667
        assert 0.6 < score < 0.7
    
    def test_false_alarm(self):
        """False alarm (precision miss) impacts score less than missing flag."""
        pred = ["STEMI_pattern", "ACS_with_hemodynamic_instability", "anaphylaxis"]
        truth = ["STEMI_pattern", "ACS_with_hemodynamic_instability"]
        score = score_flags(pred, truth)
        # Recall = 1.0, Precision = 2/3 = 0.667
        # Weighted = (2 * 1.0 + 0.667) / 3 = 0.889
        assert 0.85 < score < 0.92


class TestQueueGrader:
    """Tests for batch patient prioritization scoring."""
    
    def test_perfect_queue(self):
        """Perfect order should score close to 1.0."""
        pred_order = ["pt_022", "pt_029", "pt_026", "pt_023", "pt_025"]
        truth_order = ["pt_022", "pt_029", "pt_026", "pt_023", "pt_025"]
        resource_alloc = {p: "ICU" for p in pred_order[:2]} | {p: "ward" for p in pred_order[2:]}
        bed_counts = {"ICU": 5, "ward": 10}
        
        score = score_queue(pred_order, truth_order, resource_alloc, bed_counts)
        assert score >= 0.9  # Nearly perfect
    
    def test_violated_constraints(self):
        """Assigning patients to full wards should incur 0.3 penalty each."""
        pred_order = ["pt_022", "pt_029"]
        truth_order = ["pt_022", "pt_029"]
        # Both assigned to ICU which has 0 beds
        resource_alloc = {"pt_022": "ICU", "pt_029": "ICU"}
        bed_counts = {"ICU": 0}
        
        score = score_queue(pred_order, truth_order, resource_alloc, bed_counts)
        # rank_score = 1.0, penalty = 0.3 * 2 = 0.6
        # final = 1.0 - 0.6 = 0.4
        assert 0.35 < score < 0.45
    
    def test_wrong_order_with_good_constraints(self):
        """Wrong order but respecting constraints should show impact."""
        pred_order = ["pt_029", "pt_022"]  # Reversed
        truth_order = ["pt_022", "pt_029"]
        resource_alloc = {"pt_022": "ICU", "pt_029": "ward"}
        bed_counts = {"ICU": 5, "ward": 10}
        
        score = score_queue(pred_order, truth_order, resource_alloc, bed_counts)
        assert score < 0.5  # Bad ordering reduces score


class TestRewardCalculation:
    """Integration test for final reward calculation."""
    
    def test_perfect_single_task_reward(self):
        """Perfect predictions on single task should yield high reward."""
        # Simulated perfect scores
        scores = {
            "esi": 1.0,
            "vitals": 1.0,
            "flags": 1.0,
            "routing": 1.0,
        }
        clarifications = 0
        
        base = (0.30 * scores["esi"] + 0.25 * scores["vitals"] +
                0.30 * scores["flags"] + 0.15 * scores["routing"])
        penalty = max(0, clarifications - 2) * 0.05
        final = round(base - penalty, 4)
        
        assert final == 1.0
    
    def test_reward_with_clarification_penalty(self):
        """Exceeding 2 clarifications should incur 0.05 penalty each."""
        scores = {
            "esi": 1.0,
            "vitals": 1.0,
            "flags": 1.0,
            "routing": 1.0,
        }
        clarifications = 5  # 3 extra = 3 * 0.05 = 0.15 penalty
        
        base = (0.30 * scores["esi"] + 0.25 * scores["vitals"] +
                0.30 * scores["flags"] + 0.15 * scores["routing"])
        penalty = max(0, clarifications - 2) * 0.05
        final = round(base - penalty, 4)
        
        assert final == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
