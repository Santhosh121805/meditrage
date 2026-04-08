"""
Task 1: Single Clear - Easy difficulty
Unambiguous patient presentations requiring basic ESI triage.
"""

TASK_CONFIG = {
    "id": "task1_single_clear",
    "difficulty": "easy",
    "n_episodes": 10,
    "max_steps": 1,
    "reward_range": [0.0, 1.0],
    "description": """
Task 1: Single Clear Presentation
===================================
You receive ONE patient note with a textbook, unambiguous clinical presentation.
ESI levels are clearly indicated by vital signs and clinical findings.

Examples:
- STEMI with ST elevation
- Acute ischemic stroke (FAST positive)
- Anaphylaxis with airway compromise
- Septic shock
- Penetrating trauma

Your job: Output ESI level, extracted vitals, red flags, and recommended route.
All cases have clear correct answers. This task tests accurate data extraction.
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
]

# Standard bed counts (no resource constraints in Task 1)
BED_COUNTS = {ward: 10 for ward in AVAILABLE_WARDS}
