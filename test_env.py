"""Test environment functionality"""
from medtriage_env import MedTriageEnv
from models.schemas import TriageAction, Vitals

env = MedTriageEnv()

# Test Task 1
print("Testing Task 1 (Single Clear)...")
obs = env.reset("task1_single_clear")
print(f"  ✓ Observation type: {type(obs).__name__}")
print(f"  ✓ Note length: {len(obs.note)} characters")

action = TriageAction(
    esi_level=env.current_episode.ground_truth.esi_level,
    vitals=env.current_episode.ground_truth.vitals,
    red_flags=env.current_episode.ground_truth.red_flags,
    route_to=env.current_episode.ground_truth.correct_route,
    reasoning="Perfect prediction for testing."
)
reward, done, info = env.step(action)
print(f"  ✓ Reward (perfect): {reward}")
print(f"  ✓ Done: {done}")

# Test Task 2
print("\nTesting Task 2 (Ambiguous)...")
obs = env.reset("task2_ambiguous")
print(f"  ✓ Observation type: {type(obs).__name__}")
action = TriageAction(
    esi_level=2,
    vitals=Vitals(heart_rate=100),
    red_flags=[],
    route_to=obs.available_wards[0],
    reasoning="Ambiguous diagnosis requires further workup."
)
reward, done, info = env.step(action)
print(f"  ✓ Reward (partial): {reward}")

# Test Task 3
print("\nTesting Task 3 (Batch Constrained)...")
obs = env.reset("task3_batch_constrained")
print(f"  ✓ Observation type: {type(obs).__name__}")
print(f"  ✓ Number of patients: {len(obs.notes)}")
print(f"  ✓ Available wards: {len(obs.available_wards)}")

print("\n✓ All environment tests passed!")
