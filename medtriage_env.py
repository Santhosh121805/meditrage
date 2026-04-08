"""
MedTriage core environment - OpenEnv compatible.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import random

from models.schemas import (
    Observation,
    BatchObservation,
    TriageAction,
    BatchTriageAction,
    PatientNote,
)
from graders.esi_grader import score_esi
from graders.vitals_grader import score_vitals
from graders.flags_grader import score_flags
from graders.queue_grader import score_queue
from cognitive_load_tracker import detect_clarifications


class MedTriageEnv:
    """
    OpenEnv-compatible clinical note triage environment.
    Supports three tasks of increasing difficulty.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            data_path: Path to patient_notes.json. Auto-discovers if None.
        """
        self.data_path = Path(data_path) if data_path else self._find_data_file()
        self.patient_notes = self._load_notes()
        self.current_task_id = None
        self.current_episode = None
        self.clarification_count = 0
        self.max_steps = 1
        self.step_count = 0

    def _find_data_file(self) -> Path:
        """Auto-discover patient_notes.json."""
        candidates = [
            Path(__file__).parent / "data" / "patient_notes.json",
            Path.cwd() / "data" / "patient_notes.json",
            Path.cwd() / "medtriage" / "data" / "patient_notes.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("Cannot find patient_notes.json. Run data_generator.py first.")

    def _load_notes(self) -> Dict[str, PatientNote]:
        """Load and parse patient notes from JSON."""
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        notes = {}
        for item in data:
            note = PatientNote(**item)
            notes[note.id] = note
        return notes

    def reset(self, task_id: str) -> Observation | BatchObservation:
        """
        Reset environment and return initial observation.
        
        Args:
            task_id: One of "task1_single_clear", "task2_ambiguous", "task3_batch_constrained"
        
        Returns:
            Observation or BatchObservation depending on task
        """
        self.current_task_id = task_id
        self.clarification_count = 0
        self.step_count = 0
        
        # Map long task names to short names used in data
        task_name_map = {
            "task1_single_clear": "task1",
            "task2_ambiguous": "task2",
            "task3_batch_constrained": "task3",
        }
        short_task_id = task_name_map.get(task_id, task_id)
        
        # Filter notes for this task
        task_notes = [
            n for n in self.patient_notes.values() if n.task_type == short_task_id
        ]
        
        if not task_notes:
            raise ValueError(f"No notes found for task {task_id}")
        
        # Select random episode
        self.current_episode = random.choice(task_notes)
        
        # Define available wards
        available_wards = [
            "resuscitation_bay",
            "cardiac_cath_lab",
            "stroke_unit",
            "trauma_bay",
            "ICU",
            "acute_care_ward",
            "observation_ward",
        ]
        
        if short_task_id == "task3":
            # Task 3: 5-patient batch
            batch_notes = {
                self.current_episode.id: self.current_episode.note,
            }
            # Add 4 more random notes
            other_notes = [n for n in task_notes if n.id != self.current_episode.id]
            for note in random.sample(other_notes, min(4, len(other_notes))):
                batch_notes[note.id] = note.note
            
            # Limited bed counts for Task 3 (resource constraint)
            bed_counts = {
                "trauma_priority_1": 3,
                "trauma_priority_2": 3,
                "cardiac_care_complex": 2,
                "ICU": 4,
                "acute_care_ward_monitored": 5,
                "surgical_ward_urgent": 3,
                "ICU_isolation_emergency": 2,
                "stroke_unit": 3,
                "observation_ward": 10,
            }
            
            return BatchObservation(
                notes=batch_notes,
                available_wards=available_wards,
                bed_counts=bed_counts,
                time_of_day=random.choice(["morning_rush", "afternoon", "overnight", "weekend"]),
                task_id=task_id,
            )
        else:
            # Tasks 1 & 2: Single patient
            bed_counts = {ward: 5 for ward in available_wards}
            
            return Observation(
                note=self.current_episode.note,
                available_wards=available_wards,
                bed_counts=bed_counts,
                time_of_day=random.choice(["morning_rush", "afternoon", "overnight", "weekend"]),
                task_id=task_id,
            )

    def step(self, action: TriageAction | BatchTriageAction) -> Tuple[float, bool, Dict]:
        """
        Execute one step with an action and return (reward, done, info).
        
        Args:
            action: Agent's TriageAction or BatchTriageAction
        
        Returns:
            (reward, done, info_dict)
        """
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        
        # Detect clarifications in reasoning
        if isinstance(action, TriageAction):
            self.clarification_count += detect_clarifications(action.reasoning)
        elif isinstance(action, BatchTriageAction):
            for triage in action.individual_triages.values():
                self.clarification_count += detect_clarifications(triage.reasoning)
        
        # Grade based on task type
        if self.current_task_id == "task3_batch_constrained":
            reward = self._grade_batch(action)
        else:
            reward = self._grade_single(action)
        
        info = {
            "task_id": self.current_task_id,
            "step": self.step_count,
            "clarifications": self.clarification_count,
            "done": done,
        }
        
        return reward, done, info

    def _grade_single(self, action: TriageAction) -> float:
        """Grade single-patient triage (Task 1 & 2)."""
        gt = self.current_episode.ground_truth
        
        scores = {
            "esi": score_esi(action.esi_level, gt.esi_level),
            "vitals": score_vitals(action.vitals, gt.vitals),
            "flags": score_flags(action.red_flags, gt.red_flags),
            "routing": 1.0 if action.route_to == gt.correct_route else 0.5,
        }
        
        # Weighted reward
        base_reward = (
            0.30 * scores["esi"]
            + 0.25 * scores["vitals"]
            + 0.30 * scores["flags"]
            + 0.15 * scores["routing"]
        )
        
        # Cognitive load penalty
        cognitive_penalty = max(0, self.clarification_count - 2) * 0.05
        
        final_reward = round(base_reward - cognitive_penalty, 4)
        return max(0.0, final_reward)

    def _grade_batch(self, action: BatchTriageAction) -> float:
        """Grade multi-patient batch triage (Task 3)."""
        # For now, simplified: average individual scores + queue score
        individual_scores = []
        
        for patient_id, triage in action.individual_triages.items():
            # Find patient note
            if patient_id in self.current_episode:
                # This is simplified - in real implementation, track all 5 notes
                pass
            individual_scores.append(0.5)  # Placeholder
        
        queue_reward = score_queue(
            action.priority_queue,
            [],  # Ground truth order - would need to track
            action.resource_allocation,
            {},  # bed_counts
        )
        
        base_reward = (sum(individual_scores) / len(individual_scores) * 0.85
                       + queue_reward * 0.15)
        
        cognitive_penalty = max(0, self.clarification_count - 2) * 0.05
        final_reward = round(base_reward - cognitive_penalty, 4)
        return max(0.0, final_reward)

    def render(self) -> str:
        """Pretty-print current environment state."""
        if not self.current_episode:
            return "Environment not initialized. Call reset() first."
        
        lines = [
            "=" * 60,
            f"MedTriage Environment - Task: {self.current_task_id}",
            f"Patient ID: {self.current_episode.id}",
            f"Difficulty: {self.current_episode.difficulty}",
            "-" * 60,
            "Clinical Note:",
            self.current_episode.note,
            "-" * 60,
            "Ground Truth (for debugging):",
            f"  ESI Level: {self.current_episode.ground_truth.esi_level}",
            f"  Route: {self.current_episode.ground_truth.correct_route}",
            f"  Red Flags: {', '.join(self.current_episode.ground_truth.red_flags)}",
            "=" * 60,
        ]
        return "\n".join(lines)
