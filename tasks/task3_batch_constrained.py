"""
Task 3: Batch Constrained - Hard difficulty
Multi-patient prioritization under resource constraints.
"""

TASK_CONFIG = {
    "id": "task3_batch_constrained",
    "difficulty": "hard",
    "n_episodes": 10,
    "max_steps": 1,
    "reward_range": [0.0, 1.0],
    "description": """
Task 3: Batch Constrained Prioritization
==========================================
You receive FIVE patient notes arriving simultaneously to a busy ED.
Resources are SEVERELY LIMITED:
  - Only 3 trauma bays available
  - Only 2 ICU beds available
  - Mixed ESI levels (1-4) among the cohort

Your job:
1. Triage each patient individually (ESI, vitals, flags)
2. Create a PRIORITY QUEUE (rank 1 = most urgent)
3. Allocate each patient to a specific bed/bay
4. Respect resource constraints (don't assign >capacity)

Scoring:
- Individual triage accuracy: 85%
- Queue ordering (Kendall's tau): 10%
- Resource constraint satisfaction: 5%
- Resource violation penalty: -0.3 per violation

This task tests both clinical reasoning AND resource management.
""",
}

# Limited wards with bed constraints
AVAILABLE_WARDS = [
    "trauma_priority_1",
    "trauma_priority_2",
    "cardiac_care_complex",
    "ICU",
    "acute_care_ward_monitored",
    "surgical_ward_urgent",
    "ICU_isolation_emergency",
    "stroke_unit",
    "observation_ward",
]

# CONSTRAINED bed counts - induces real resource decisions
BED_COUNTS = {
    "trauma_priority_1": 3,
    "trauma_priority_2": 3,
    "cardiac_care_complex": 2,
    "ICU": 2,  # Very limited
    "acute_care_ward_monitored": 5,
    "surgical_ward_urgent": 3,
    "ICU_isolation_emergency": 2,
    "stroke_unit": 3,
    "observation_ward": 10,
}
