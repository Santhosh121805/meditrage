"""Quick validation script"""
import json

data = json.load(open('data/patient_notes.json'))
print(f'✓ Generated {len(data)} patient notes')
print(f'  Task 1 (easy): {sum(1 for n in data if n["task_type"] == "task1")}')
print(f'  Task 2 (medium): {sum(1 for n in data if n["task_type"] == "task2")}')
print(f'  Task 3 (hard): {sum(1 for n in data if n["task_type"] == "task3")}')
print(f'\n✓ Sample record:')
print(f'  ID: {data[0]["id"]}')
print(f'  ESI Level: {data[0]["ground_truth"]["esi_level"]}')
print(f'  Red Flags: {len(data[0]["ground_truth"]["red_flags"])}')
print(f'  Route: {data[0]["ground_truth"]["correct_route"]}')
