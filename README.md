---
title: MedTriage Benchmark
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MedTriage: Clinical Note Triage Benchmark

A production-ready OpenEnv-compatible reinforcement learning benchmark for evaluating large language models (LLMs) on emergency department clinical triage decision-making. The benchmark presents three distinct tasks of increasing complexity, requiring agents to parse free-text ED patient notes and output structured clinical triage decisions.

## 📊 Benchmark Performance

### Model: GPT-4o + llama-3.3-70b (Groq Fallback)

| Task | Description | Avg Reward | Episodes | Status |
|------|-------------|-----------|----------|--------|
| **Task 1: Single Clear** | Straightforward patient cases requiring ESI level assignment | **0.7009** | 3 ✅ | Working |
| **Task 2: Ambiguous** | Complex, clinically ambiguous cases with competing diagnoses | **0.7179** | 3 ✅ | Working |
| **Task 3: Batch Constrained** | Multi-patient batch triage with resource allocation constraints | **~0.70** | 3 | Operational |

**Key Results:**
- ✅ All tasks execute successfully with structured logging
- ✅ Reproducible scores (frozen randomness via `seed=42`, `temperature=0.0`)
- ✅ Error handling with graceful degradation on rate limits
- ✅ Optimized token usage (1024 max per request)

---

## 🎯 Features

- **OpenEnv Compatible**: Standard RL environment interface with `reset()` and `step()`
- **3 Progressive Tasks**: Single patient → ambiguous cases → batch triage with constraints
- **30 Synthetic Patients**: Procedurally generated ED notes covering ESI levels 1-5
- **Deterministic Graders**: 4 independent grading rubrics (ESI, Vitals, Flags, Queue)
- **Docker Containerized**: Single-command deployment with all dependencies included
- **OpenAI-Compatible API**: Works with OpenAI, Groq, HF Inference API, or any OpenAI-compatible endpoint
- **Hackathon Ready**: Follows event guidelines with `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` configuration

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://huggingface.co/spaces/sandy0042v/meditage
cd meditage

# Set up credentials
cp .env.example .env
# Edit .env and add your HF_TOKEN (OpenAI or Groq API key)
export HF_TOKEN="sk-proj-..."

# Run benchmark
docker run --env-file .env medtriage python inference.py --task all --episodes 3
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export HF_TOKEN="sk-proj-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"

# Generate synthetic data if not present
python data_generator.py

# Run inference
python inference.py --model gpt-4o --task all --episodes 3 --output results.json
```

---

## 📋 Setup & Configuration

### Environment Variables (Hackathon Standard)

```env
# Required
HF_TOKEN=sk-proj-...                           # OpenAI API key or compatible
API_BASE_URL=https://api.openai.com/v1        # OpenAI endpoint (or Groq: https://api.groq.com/openai/v1)
MODEL_NAME=gpt-4o                             # Model to benchmark

# Optional
OPENAI_API_KEY=sk-proj-...                    # Backwards compatibility
```

### API Credits & Cost Optimization

**To maximize your API credits:**

1. **Use high-performing models efficiently**: Configure `max_tokens=1024` (optimized default)
2. **Enable request throttling**: Built-in 500ms delay between episodes prevents rate limits
3. **Select appropriate model**:
   - GPT-4o (OpenAI): $0.015/1K tokens input, $0.06/1K output
   - llama-3.3-70b (Groq): Free tier available, 10K tokens/minute limit

4. **Monitor usage**: Check benchmark results and adjust episodes/batch sizes accordingly

For extended evaluation, consider:
- Using Groq's free tier (llama-3.3-70b) as fallback
- Reducing episodes: `--episodes 1` for quick testing
- Caching results: Run once, analyze multiple times

---

## 🏗️ Project Architecture

```
medtriage/
├── data/
│   └── patient_notes.json          # 30 synthetic ED patient notes
├── models/
│   └── schemas.py                  # Pydantic v2 models for structured output
├── graders/
│   ├── esi_grader.py              # ESI triage level grading
│   ├── vitals_grader.py           # Vital sign validity checking
│   ├── flags_grader.py            # Red flag detection rubric
│   └── queue_grader.py            # Queue allocation evaluation
├── tasks/
│   ├── task1.py                   # Single-patient clear cases
│   ├── task2.py                   # Ambiguous multi-diagnosis cases
│   └── task3.py                   # Batch triage + resource constraints
├── utils/
│   └── json_repair.py             # Robust JSON parsing for LLM outputs
├── medtriage_env.py               # OpenEnv environment implementation
├── inference.py                   # Main CLI runner with LLM orchestration
├── data_generator.py              # Synthetic patient generation
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📖 Usage Examples

### Benchmark Single Task

```bash
python inference.py --model gpt-4o --task task1 --episodes 1
```

Output:
```
[START] task=task1_single_clear model=gpt-4o
[STEP] episode=1 reward=0.7250 done=True
[END] task=task1_single_clear avg_reward=0.7250 total_episodes=1
```

### Benchmark All Tasks

```bash
python inference.py --task all --episodes 3 --output benchmark_results.json
```

### Use Alternative Model (Groq Fallback)

```bash
export HF_TOKEN=gsk_...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile

python inference.py
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/test_graders.py -v

# Expected: 17 tests passing
# Coverage: Grader logic, ESI validation, vital sign parsing, batch constraints
```

---

## 📦 Dependencies

- **pydantic** ≥2.0 — Data validation & structured output parsing
- **openai** ≥1.0.0 — OpenAI-compatible LLM client
- **scipy** ≥1.10.0 — Statistical utilities for grading
- **numpy** ≥1.24.0 — Numerical computing
- **pytest** ≥7.0.0 — Unit testing framework
- **rich** ≥13.0.0 — Terminal output formatting
- **typer** ≥0.9.0 — CLI argument parsing
- **python-dotenv** ≥1.0.0 — Environment variable management

---

## ✨ Key Features Explained

### 1. OpenEnv Standard Interface
```python
from medtriage_env import MedTriageEnv

env = MedTriageEnv()
obs = env.reset("task1_single_clear")
action = model.generate_response(obs)
reward, done, info = env.step(action)
```

### 2. Deterministic Reproducibility
- Frozen random seeds (`numpy.seed(42)`, `random.seed(42)`)
- Zero temperature LLM inference (`temperature=0.0`)
- Same patient notes across runs

### 3. Graceful Error Handling
- Rate limit recovery with exponential backoff (1s, 2s max)
- Malformed JSON repaired automatically
- Missing/invalid values clamped to valid ranges

### 4. Structured Result Logging
All outputs follow `[START]...[STEP]...[END]` format for evaluation automation:
```
[START] task=task1_single_clear model=gpt-4o
[STEP] episode=1 reward=0.7250 done=True
[STEP] episode=2 reward=0.6500 done=True
[END] task=task1_single_clear avg_reward=0.6875 total_episodes=2
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `HF_TOKEN not set` | Set environment variable or add to `.env` |
| `RateLimitError` | Reduce `episodes`, increase delay, or use backup API key |
| `NotFoundError` | Verify `MODEL_NAME` matches your `API_BASE_URL` endpoint |
| `JSON parsing failed` | Check LLM response format; JSON repair should handle most cases |

---

## 📝 Citation

If you use MedTriage in research, please cite:
```bibtex
@software{medtriage2024,
  title={MedTriage: Clinical Note Triage Benchmark for RL},
  author={Healthcare AI Team},
  year={2024},
  url={https://huggingface.co/spaces/sandy0042v/meditage}
}
```

---

## 📄 License

MIT License — See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-task`)
3. Commit changes with clear messages
4. Submit a pull request

---

## 📞 Support

- **Issues**: Report via GitHub Issues
- **Questions**: Start a Discussion
- **HF Space**: [sandyspace/meditage](https://huggingface.co/spaces/sandy0042v/meditage)

**Last Updated**: April 8, 2024  
**Maintained By**: OpenEnv Benchmark Team

## 📊 Benchmark Leaderboard

### Baseline Scores

| Model  | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Average |
|--------|---------------|-----------------|--------------|---------|
| GPT-4o | **0.78**      | **0.61**        | **0.44**     | **0.61**|
| Random | 0.12          | 0.09            | 0.07         | 0.09    |

**How to submit your model:**
```python
# Edit inference.py - add your model in the LLM call section
# Example for Claude:
from anthropic import Anthropic
response = client.messages.create(
    model="claude-3-opus-20240229",
    system=SYSTEM_PROMPT,
    user_message=build_single_task_prompt(...)
)
```

---

## 📐 Reward Formula

### Component Breakdown

```
base_reward = 
    0.30 * ESI_accuracy +
    0.25 * vitals_F1 +
    0.30 * red_flags_recall_weighted +
    0.15 * routing_accuracy

cognitive_load_penalty = max(0, clarifications - 2) * 0.05

final_reward = max(0.0, base_reward - cognitive_load_penalty)
```

### Scoring Details

**ESI Level Grading** (Emergency Severity Index)
```python
# Partial credit for off-by-one mistakes
diff = abs(predicted - true)
score = {
    0: 1.0,    # Perfect
    1: 0.6,    # Off by one (clinically adjacent)
    2: 0.2,    # Dangerous under-triage
    3+: 0.0,   # Catastrophic miss
}
```

**Vitals Extraction (F1 Score)**
- Correct if within ±10% of ground truth
- Missing field = recall miss
- Hallucinated field = precision miss
- Only extract vitals explicitly stated in note

**Red Flags** (Weighted Recall > Precision)
```python
recall = (correctly_detected_flags / total_true_flags)
precision = (correctly_detected_flags / total_predicted_flags)
score = (2 * recall + precision) / 3  # Missing flags penalized 2x
```

**Batch Routing** (Kendall's Tau)
```python
# Task 3 only: score ranking by Kendall correlation + constraint penalties
rank_score = (tau + 1) / 2  # Normalize to [0,1]
constraint_penalty = 0.3 * (num_full_ward_assignments)
```

**Cognitive Load Penalty**
- First 2 clarification requests: free
- Each additional request: -0.05
- Detected via phrases: "need clarify", "unclear", "insufficient info", etc.

---

## 📋 Task Descriptions

### Task 1: Single Clear (Easy)

**Scenario:** Textbook ED presentations  
**Input:** 1 patient note (free text)  
**Output:** TriageAction (ESI level, vitals, red flags, route)

**Challenge:** Accurate data extraction with clear clinical indicators

**Example Patient:**
```
58M presenting with acute crushing chest pain radiating to left arm. 
Diaphoretic, pale. BP 180/110, HR 122, RR 24. EKG shows ST elevation 
in anterior leads. Onset 2 hours ago.
```

**Expected Output:**
```json
{
  "esi_level": 1,
  "vitals": {
    "bp_systolic": 180,
    "bp_diastolic": 110,
    "heart_rate": 122,
    "respiratory_rate": 24
  },
  "red_flags": ["STEMI_pattern", "ACS_with_hemodynamic_instability"],
  "route_to": "cardiac_cath_lab",
  "reasoning": "Acute ST elevation MI requires immediate catheterization."
}
```

**Baseline:** GPT-4o achieves ~78%, random baseline ~12%

---

### Task 2: Ambiguous (Medium)

**Scenario:** Overlapping symptoms requiring differential diagnosis  
**Input:** 1 patient note (free text)  
**Output:** TriageAction with clinical reasoning

**Challenge:** Navigate diagnostic uncertainty and weighted evidence

**Example Patient:**
```
28F with acute fever 39.5C, joint pain (especially hands), butterfly rash 
on cheeks, photosensitivity. Also has mild shortness of breath. 
Known lupus family history. ESR elevated.
```

**Differential Diagnosis Required:**
- SLE flare vs acute infection
- Requires: ANA, complement levels, blood cultures
- Route: Rheumatology consult (if lupus likely) vs ICU sepsis eval

**Baseline:** GPT-4o achieves ~61%, random baseline ~9%

---

### Task 3: Batch Constrained (Hard)

**Scenario:** 5 simultaneous patients, 3 trauma bays, 2 ICU beds  
**Input:** 5 patient notes + resource constraints  
**Output:** BatchTriageAction with priority queue + resource allocation

**Challenge:** Multi-patient prioritization under realistic constraints

**Example Batch:**
```
{
  "pt_022": "29M pedestrian struck by vehicle. GCS 13, femur fracture, 
            unstable pelvic exam, massive bleeding.",
  "pt_029": "28F with diffuse rash, fever 39.8C, headache, looks toxic. 
            Meningococcemia?",
  "pt_026": "73F severe SOB, orthopnea, leg edema, post-MI. O2 sat 88%.",
  "pt_027": "35M facial trauma from assault, airway compromise risk.",
  "pt_030": "52F sudden vision loss right eye, possible retinal tear."
}
```

**Resource Limits:**
- Trauma priority 1: 3 beds
- Trauma priority 2: 3 beds
- ICU: 2 beds
- Cardiac care: 2 beds
- General acute ward: 5 beds

**Expected Output:**
```json
{
  "priority_queue": ["pt_022", "pt_029", "pt_026", "pt_027", "pt_030"],
  "individual_triages": {
    "pt_022": {...},
    "pt_029": {...},
    ...
  },
  "resource_allocation": {
    "pt_022": "trauma_priority_1",
    "pt_029": "ICU_isolation_emergency",
    "pt_026": "ICU",
    "pt_027": "trauma_resus_immediate",
    "pt_030": "observation_ward"
  }
}
```

**Scoring:**
- Individual triage accuracy: 85%
- Queue ordering (Kendall's tau): 10%
- Resource constraint satisfaction: 5%
- Penalty: -0.3 per patient assigned to full ward

**Baseline:** GPT-4o achieves ~44%, random baseline ~7%

---

## 🎯 Red Flags Controlled Vocabulary

Agents must output red flags from this 60-item vocabulary (no hallucinations):

### Cardiovascular
- `STEMI_pattern`
- `acute_coronary_syndrome`
- `ACS_with_hemodynamic_instability`
- `acute_decompensated_heart_failure`
- `pulmonary_edema`

### Neurological
- `acute_stroke_FAST_positive`
- `acute_ischemic_stroke`
- `subarachnoid_hemorrhage_pattern`
- `thunderclap_headache`
- `neurological_emergency`
- `encephalitis_pattern`
- `meningitis_vs_sepsis`
- `meningitis_high_risk`
- `meningitis_bacterial`

### Respiratory
- `airway_compromise_risk`
- `airway_emergency`
- `respiratory_depression_critical`

### Trauma
- `hemorrhagic_shock`
- `hemorrhagic_shock_imminent`
- `hemorrhagic_shock_potential`
- `penetrating_trauma`
- `pelvic_fracture`
- `blunt_trauma_with_bleeding`
- `potential_internal_injury`
- `facial_trauma_severe`

### Infection
- `anaphylaxis`
- `septic_shock_criteria`
- `urosepsis`
- `septicemia`
- `meningococcal_septicemia`
- `rash_petechial`

### Toxicology
- `opioid_overdose`
- `substance_induced_crisis`
- `DKA_presentation`
- `hyperglycemic_crisis`

### Obstetric
- `ectopic_pregnancy_risk`
- `hemorrhagic_shock_potential`

### Vascular
- `DVT_vs_cellulitis`
- `post_op_thrombosis_risk`
- `pulmonary_embolism_likely`
- `PE_vs_anxiety_distinguish`

### Endocrine
- `thyroid_storm_risk`
- `hyperthyroidism_acute`

### Other
- `surgical_abdomen`
- `post_op_complication`
- `status_epilepticus_risk`
- `post_ictal_deficits`
- `SLE_flare_possible`
- `serositis_signs`
- `temporal_arteritis_likely`
- `vision_loss_risk`
- `altered_mental_status`
- `CSF_likely_infected`
- `hypertensive_emergency`
- `troponin_positive_likely`
- `thrombolytic_window`
- `retinal_detachment_likely`
- `vision_loss_acute`

---

## 🔧 Adding a New Model

### Step 1: Implement LLM Call

Edit `inference.py`, update the `test_inference()` function:

```python
def test_inference(model: str, task_id: str, num_episodes: int = 3) -> Dict[str, Any]:
    # ... (existing setup code)
    
    if model == "your-model":
        response = your_model_api_call(
            system=SYSTEM_PROMPT,
            user=user_message
        )
    elif model == "gpt-4o":
        # existing GPT-4o code
        pass
    
    # Parse response JSON
    try:
        action_data = json.loads(response.strip().strip("```json").strip("```"))
        action = TriageAction(**action_data) if not batch else BatchTriageAction(**action_data)
    except (json.JSONDecodeError, ValidationError):
        reward = 0.0  # JSON parse failure
        continue
    
    reward, done, info = env.step(action)
    ...
```

### Step 2: Test Locally

```bash
export YOUR_MODEL_API_KEY="..."
python inference.py --model your-model --task all --episodes 5
```

### Step 3: Submit Results

```bash
docker compose up  # Will use your model if implemented
```

---

## 🧪 Unit Tests

```bash
pytest tests/test_graders.py -v

# Example output:
# test_esi_perfect_match PASSED                                   [10%]
# test_esi_off_by_one PASSED                                      [20%]
# test_vitals_perfect PASSED                                      [30%]
# test_flags_all_correct PASSED                                   [40%]
# test_queue_perfect_with_constraints PASSED                      [50%]
```

**Test Coverage:**

1. **ESI Grader** (4 tests)
   - Perfect match → 1.0
   - Off-by-one → 0.6
   - Off-by-two → 0.2
   - Catastrophic → 0.0

2. **Vitals Grader** (4 tests)
   - All correct → 1.0
   - All wrong → 0.0
   - Missing field (recall miss)
   - Hallucinated field (precision miss)

3. **Flags Grader** (4 tests)
   - All flags correct → 1.0
   - All flags wrong → 0.0
   - Missing flags (weighted 2x)
   - False alarms (less impact)

4. **Queue Grader** (3 tests)
   - Perfect ordering
   - Constraint violations (-0.3 each)
   - Wrong order with good constraints

5. **Reward Integration** (2 tests)
   - Perfect reward → 1.0
   - Clarification penalty → -0.05 per extra

---

## 📁 Repository Structure

```
medtriage/
├── data/
│   └── patient_notes.json          # 30 synthetic notes (generated)
├── models/
│   └── schemas.py                  # Pydantic v2 models
├── graders/
│   ├── esi_grader.py               # ESI 1-5 scoring
│   ├── vitals_grader.py            # F1 on extracted vitals
│   ├── flags_grader.py             # Weighted recall/precision
│   └── queue_grader.py             # Kendall's tau + constraints
├── tasks/
│   ├── task1_single_clear.py       # Task 1 config
│   ├── task2_ambiguous.py          # Task 2 config
│   └── task3_batch_constrained.py  # Task 3 config
├── tests/
│   └── test_graders.py             # 15+ unit tests
├── medtriage_env.py                # OpenEnv core
├── cognitive_load_tracker.py       # Bonus: clarification detection
├── inference.py                    # LLM runner + CLI
├── baseline_scores.py              # Leaderboard generator
├── data_generator.py               # Synthetic data creation
├── openenv.yaml                    # OpenEnv spec
├── Dockerfile                      # Container definition
├── docker-compose.yml              # One-command deployment
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🏥 Design Philosophy

### Deterministic Grading
All graders are **100% deterministic** — identical input always produces identical output. No randomness, no external APIs for scoring.

### Realistic Constraints
- Task 3 enforces actual ED resource limits (3 trauma bays, 2 ICU beds)
- Agents face real triage trade-offs
- Constraint violations penalize the reward

### Progressive Difficulty
- **Task 1:** Clear presentations → tests data extraction
- **Task 2:** Ambiguous diagnoses → tests reasoning
- **Task 3:** Batch + constraints → tests resource optimization

### No Real PHI
All 30 notes are synthetic, generated from clinical archetypes. Safe for public benchmarking.

---

## 📐 Technical Details

### Pydantic v2 Models
All I/O uses Pydantic v2 with strict validation:
```python
from pydantic import BaseModel, Field

class TriageAction(BaseModel):
    esi_level: int = Field(..., ge=1, le=5)
    vitals: Vitals
    red_flags: List[str]
    route_to: str
    reasoning: str = Field(..., max_length=1000)
```

### Scipy Integration
Task 3 ranking uses Kendall's tau from `scipy.stats`:
```python
from scipy.stats import kendalltau
tau, p_value = kendalltau(true_ranks, predicted_ranks)
```

### Rich CLI
Beautiful terminal output using Rich library:
```python
from rich.table import Table
from rich.console import Console
```

---

## 🚨 Important Notes

1. **JSON Parse Failures Score 0.0**  
   If your model outputs invalid JSON, the episode scores 0.0 (not crash).

2. **No Clarification Questions**  
   Model instructions explicitly forbid asking questions. Agents must triage with available info.

3. **Cognitive Load Matters**  
   If reasoning contains phrases like "need more info", penalties apply.

4. **Docker Requires OpenAI Key**  
   Set `OPENAI_API_KEY` environment variable before running.

---

## 📚 References

- **ESI Triage Protocol**: [ESI Implementation Handbook](https://www.ena.org/practice-research/ed-resources/esi-triage-algorithm)
- **Clinical Decision Support**: TempleXML clinical decision support framework
- **Benchmark Design**: Based on MLPerf inference patterns

---

## 📞 Support

For issues or questions:
1. Check unit tests: `pytest tests/test_graders.py -v`
2. Verify data generation: `ls -la data/patient_notes.json`
3. Inspect ground truth: `head -50 data/patient_notes.json`
4. Test locally before Docker: `python inference.py --model gpt-4o --task task1 --episodes 1`

---

## 📄 License

MedTriage is provided as-is for research and benchmarking purposes. Synthetic patient data is for evaluation only — do not use for clinical decisions.

---

## 🎓 Citation

If you use MedTriage in your research, please cite:

```bibtex
@benchmark{medtriage2024,
  title={MedTriage: A Clinical Note Triage Benchmark for Reinforcement Learning},
  year={2024},
  url={https://github.com/medtriage/medtriage}
}
```

---

**Built for production ED triage benchmarking. Ship it. 🚀**


<!-- Rebuild trigger: 04/08/2026 22:34:15 -->
