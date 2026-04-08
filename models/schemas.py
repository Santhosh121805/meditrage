"""
MedTriage Pydantic v2 models for all I/O structures.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Vitals(BaseModel):
    """Extracted vital signs from patient note."""
    bp_systolic: Optional[int] = Field(None, ge=40, le=280)
    bp_diastolic: Optional[int] = Field(None, ge=20, le=200)
    heart_rate: Optional[int] = Field(None, ge=20, le=240)
    spo2: Optional[float] = Field(None, ge=30, le=100)
    temperature: Optional[float] = Field(None, ge=68, le=112)
    respiratory_rate: Optional[int] = Field(None, ge=5, le=100)
    gcs: Optional[int] = Field(None, ge=3, le=15)


class TriageAction(BaseModel):
    """Output for single-patient triage task (Task 1 & 2)."""
    esi_level: int = Field(..., ge=1, le=5)
    vitals: Vitals
    red_flags: List[str] = Field(default_factory=list)
    route_to: str
    reasoning: str = Field(..., max_length=1000)


class BatchTriageAction(BaseModel):
    """Output for multi-patient batch triage task (Task 3)."""
    priority_queue: List[str]
    individual_triages: Dict[str, TriageAction]
    resource_allocation: Dict[str, str]


class Observation(BaseModel):
    """State for single-patient triage task."""
    note: str
    available_wards: List[str]
    bed_counts: Dict[str, int]
    time_of_day: str
    task_id: str


class BatchObservation(BaseModel):
    """State for multi-patient batch triage task."""
    notes: Dict[str, str]
    available_wards: List[str]
    bed_counts: Dict[str, int]
    time_of_day: str
    task_id: str


class GroundTruth(BaseModel):
    """Ground truth for a patient note."""
    esi_level: int = Field(..., ge=1, le=5)
    vitals: Vitals
    red_flags: List[str]
    correct_route: str
    clinician_reasoning: str


class PatientNote(BaseModel):
    """Structure for a single patient record."""
    id: str
    note: str
    ground_truth: GroundTruth
    task_type: str  # "task1", "task2", or "task3"
    difficulty: str  # "easy", "medium", "hard"
