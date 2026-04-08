"""
Task 2: Ambiguous - Medium difficulty
Vague symptoms requiring clinical reasoning to differentiate similar conditions.
"""

TASK_CONFIG = {
    "id": "task2_ambiguous",
    "difficulty": "medium",
    "n_episodes": 10,
    "max_steps": 1,
    "reward_range": [0.0, 1.0],
    "description": """
Task 2: Ambiguous Presentations
================================
You receive ONE patient note with overlapping symptoms requiring differential diagnosis.

Examples:
- Lupus flare vs acute infection (fever + joint pain + rash)
- PE vs anxiety (dyspnea + chest pain + tachycardia)
- Ectopic pregnancy vs UTI (lower abdominal pain + spotting)
- DKA vs intoxication (fruity breath + altered mental status)
- Meningitis vs viral illness (headache + fever + neck stiffness)

Your job: Use clinical reasoning to narrow the differential, assign ESI appropriately,
and recommend the correct initial workup route.

Difficulty: Requires more sophisticated reasoning than Task 1.
Tests: Ability to weigh competing hypotheses and make sound clinical judgment.
""",
}

# Available wards for routing
AVAILABLE_WARDS = [
    "resuscitation_bay",
    "cardiac_cath_lab",
    "stroke_unit",
    "trauma_bay",
    "ICU",
    "acute_care_ward",
    "observation_ward",
    "rheumatology_consult_ward",
    "chest_imaging_ward",
    "obstetrics_emergency",
    "endocrine_care_ward",
    "vascular_surgery_urgent",
    "acute_neuro_ward",
]

# Standard bed counts (no resource constraints in Task 2)
BED_COUNTS = {ward: 10 for ward in AVAILABLE_WARDS}
