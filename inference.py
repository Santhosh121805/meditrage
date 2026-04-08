"""
MedTriage inference runner - execute LLM against all 3 tasks.
CLI: python inference.py --model gpt-4o --task all --output results.json
     python inference.py --model llama-3.3-70b-versatile --task all --output results.json
"""

import json
import argparse
import time
import os
import random
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv()

from models.schemas import TriageAction, BatchTriageAction, Observation, BatchObservation
from medtriage_env import MedTriageEnv
from utils.json_repair import robust_json_parse, clamp_vital_ranges

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

console = Console()

# ============ Flask Server Setup ============
app = Flask(__name__)

# Global state for OpenEnv API
_env = None
_client = None
_current_observation = None
_model_name = None
_task_id = None


def _get_or_init_env_client(model: str):
    """Initialize environment and client if not already done."""
    global _env, _client, _model_name
    if _env is None:
        _env = MedTriageEnv()
    if _client is None or model != _model_name:
        _client = get_llm_client(model)
        _model_name = model
    return _env, _client


@app.route("/", methods=["GET"])
def root_endpoint():
    """Root endpoint - confirms server is running."""
    return jsonify({"message": "MedTriage OpenEnv Server", "status": "running"}), 200


@app.route("/tasks", methods=["GET"])
def tasks_endpoint():
    """List available tasks."""
    return jsonify({
        "tasks": [
            "task1_single_clear",
            "task2_ambiguous", 
            "task3_batch_constrained"
        ]
    }), 200


@app.route("/reset", methods=["POST", "GET"])
@app.route("/reset/<task_id>", methods=["POST", "GET"])
def reset_endpoint(task_id=None):
    """
    Reset environment to start a new episode.
    
    Accepts task_id from multiple sources (in priority order):
    1. URL path: POST /reset/task1_single_clear
    2. Query parameter: POST /reset?task_id=task1_single_clear
    3. JSON body: POST /reset with {"task_id": "task1_single_clear"}
    """
    try:
        import sys
        print(f"[DEBUG] reset_endpoint called with path task_id={task_id}", file=sys.stderr)
        print(f"[DEBUG] Request full_path={request.full_path}", file=sys.stderr)
        print(f"[DEBUG] Request method={request.method}", file=sys.stderr)
        
        # First try to get JSON body - this is most reliable
        json_data = request.get_json(force=True, silent=True) or {}
        print(f"[DEBUG] JSON body: {json_data}", file=sys.stderr)
        
        # Try to get task_id from multiple sources
        if not task_id:
            # Try JSON body first
            task_id = json_data.get("task_id")
            if task_id:
                print(f"[DEBUG] Got task_id from JSON body: {task_id}", file=sys.stderr)
            
            # If not in body, try query params
            if not task_id:
                task_id = request.args.get("task_id")
                if task_id:
                    print(f"[DEBUG] Got task_id from query param: {task_id}", file=sys.stderr)
        else:
            print(f"[DEBUG] Got task_id from path param: {task_id}", file=sys.stderr)
        
        # Get model from body or query
        model = json_data.get("model") or request.args.get("model") or os.environ.get("MODEL_NAME", "gpt-4o")
        print(f"[DEBUG] Using model={model}", file=sys.stderr)
        
        if not task_id:
            print(f"[DEBUG] No task_id found anywhere!", file=sys.stderr)
            print(f"[DEBUG] Request content-type: {request.content_type}", file=sys.stderr)
            print(f"[DEBUG] Request data (raw): {request.data}", file=sys.stderr)
            print(f"[DEBUG] Using default task_id: task1_single_clear", file=sys.stderr)
            # Default to first task if not specified
            task_id = "task1_single_clear"
        
        print(f"[DEBUG] Initializing environment with task_id={task_id}, model={model}", file=sys.stderr)
        env, client = _get_or_init_env_client(model)
        
        # Reset environment
        global _current_observation, _task_id
        print(f"[DEBUG] Calling env.reset({task_id})", file=sys.stderr)
        _current_observation = env.reset(task_id)
        _task_id = task_id
        print(f"[DEBUG] Reset complete, observation type: {type(_current_observation)}", file=sys.stderr)
        
        # Return observation as JSON
        if isinstance(_current_observation, Observation):
            obs_dict = _current_observation.model_dump()
        else:
            obs_dict = _current_observation.model_dump()
        
        print(f"[DEBUG] Returning reset response", file=sys.stderr)
        return jsonify({
            "status": "reset",
            "task_id": task_id,
            "model": model,
            "observation": obs_dict
        }), 200
    
    except Exception as e:
        import traceback
        print(f"[ERROR] {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


@app.route("/step", methods=["POST"])
def step_endpoint():
    """
    Execute one step in the environment.
    
    Expected JSON body:
    {
        "action": { ... action data ... }
    }
    """
    try:
        import sys
        global _current_observation, _env, _client
        
        print(f"[DEBUG] /step endpoint called", file=sys.stderr)
        
        if _env is None or _current_observation is None:
            print(f"[DEBUG] Environment not initialized", file=sys.stderr)
            return jsonify({"error": "environment not initialized, call /reset first"}), 400
        
        print(f"[DEBUG] Current observation type: {type(_current_observation)}", file=sys.stderr)
        
        # Force parse JSON regardless of Content-Type header
        data = request.get_json(force=True, silent=True) or {}
        action_data = data.get("action")
        
        print(f"[DEBUG] Received action_data: {action_data is not None}", file=sys.stderr)
        
        if not action_data:
            print(f"[DEBUG] No action provided", file=sys.stderr)
            return jsonify({"error": "action required in request body"}), 400
        
        print(f"[DEBUG] Validating action as {type(_current_observation).__name__}", file=sys.stderr)
        
        try:
            # Parse and validate action based on observation type
            if isinstance(_current_observation, Observation):
                print(f"[DEBUG] Parsing as TriageAction", file=sys.stderr)
                action = TriageAction(**action_data)
            else:
                print(f"[DEBUG] Parsing as BatchTriageAction", file=sys.stderr)
                action = BatchTriageAction(**action_data)
            
            print(f"[DEBUG] Action validated successfully", file=sys.stderr)
        except Exception as validation_error:
            print(f"[DEBUG] Action validation failed: {validation_error}", file=sys.stderr)
            return jsonify({
                "error": f"action validation failed: {str(validation_error)}",
                "expected_schema": "TriageAction or BatchTriageAction"
            }), 400
        
        # Execute step
        print(f"[DEBUG] Executing step with action", file=sys.stderr)
        reward, done, info = _env.step(action)
        print(f"[DEBUG] Step complete: reward={reward}, done={done}", file=sys.stderr)
        
        # Build minimal response to avoid serialization hangs
        response_dict = {
            "status": "step",
            "reward": float(reward),
            "done": bool(done),
        }
        
        # Add info only if it's safe
        if info:
            try:
                # Only include key fields from info
                safe_info = {}
                for key, val in info.items():
                    if isinstance(val, (int, float, bool, str, type(None))):
                        safe_info[key] = val
                response_dict["info"] = safe_info
            except:
                response_dict["info"] = {}
        
        print(f"[DEBUG] Returning step response", file=sys.stderr)
        return jsonify(response_dict), 200
    
    except Exception as e:
        import traceback
        print(f"[ERROR in /step] {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/health", methods=["GET"])
def health_endpoint():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/debug", methods=["GET", "POST"])
def debug_endpoint():
    """Debug endpoint to inspect incoming requests."""
    return jsonify({
        "method": request.method,
        "path": request.path,
        "full_path": request.full_path,
        "args": dict(request.args),
        "json_body": request.get_json(force=True, silent=True),
        "headers": dict(request.headers),
        "remote_addr": request.remote_addr
    }), 200

def get_llm_client(model: str) -> OpenAI:
    """
    Initialize OpenAI-compatible client using HF_TOKEN and API_BASE_URL.
    Works with any OpenAI-compatible endpoint (OpenAI, Groq, HF Inference API, etc.)
    
    Args:
        model: Model identifier
    
    Returns:
        Configured OpenAI client
    """
    import httpx
    
    # Required: HF_TOKEN (NO DEFAULT - must be provided)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Required for OpenAI-compatible API access."
        )
    
    # Optional: API_BASE_URL with sensible default
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    
    return OpenAI(api_key=hf_token, base_url=base_url, http_client=httpx.Client())


# Red flag controlled vocabulary
RED_FLAG_VOCABULARY = [
    "STEMI_pattern",
    "acute_stroke_FAST_positive",
    "anaphylaxis",
    "septic_shock_criteria",
    "hemorrhagic_shock",
    "airway_compromise_risk",
    "subarachnoid_hemorrhage_pattern",
    "pulmonary_embolism_likely",
    "blunt_trauma_with_bleeding",
    "thunderclap_headache",
    "potential_internal_injury",
    "respiratory_depression_critical",
    "opioid_overdose",
    "status_epilepticus_risk",
    "post_ictal_deficits",
    "SLE_flare_possible",
    "serositis_signs",
    "PE_vs_anxiety_distinguish",
    "ectopic_pregnancy_risk",
    "hemorrhagic_shock_potential",
    "DKA_presentation",
    "hyperglycemic_crisis",
    "meningitis_vs_sepsis",
    "encephalitis_pattern",
    "temporal_arteritis_likely",
    "vision_loss_risk",
    "thyroid_storm_risk",
    "hyperthyroidism_acute",
    "DVT_vs_cellulitis",
    "post_op_thrombosis_risk",
    "meningitis_high_risk",
    "CSF_likely_infected",
    "hypertensive_emergency",
    "substance_induced_crisis",
    "acute_coronary_syndrome",
    "troponin_positive_likely",
    "pelvic_fracture",
    "hemorrhagic_shock_imminent",
    "urosepsis",
    "altered_mental_status",
    "meningitis_bacterial",
    "septicemia",
    "surgical_abdomen",
    "post_op_complication",
    "acute_decompensated_heart_failure",
    "pulmonary_edema",
    "airway_emergency",
    "facial_trauma_severe",
    "acute_ischemic_stroke",
    "thrombolytic_window",
    "ACS_with_hemodynamic_instability",
    "neurological_emergency",
    "meningococcal_septicemia",
    "rash_petechial",
    "retinal_detachment_likely",
    "vision_loss_acute",
]


SYSTEM_PROMPT = """You are an expert emergency department triage nurse with 15 years of experience.

Your task is to triage a patient or group of patients based on clinical presentation.

CRITICAL RULES:
1. ESI levels: 1 = immediate life threat, 2 = high-risk situation, 3-5 = lower acuity
2. Extract ONLY vitals explicitly stated in the note. Do NOT infer missing values (leave as null).
3. Red flags are from a controlled vocabulary. Output ONLY recognized codes.
4. Route patients only to wards in the available_wards list.
5. Do NOT ask clarifying questions. Make your best judgment with available info.
6. Output ONLY valid JSON matching the required schema. No markdown, no explanations.

Available Red Flag Codes:
{red_flags}

Available Wards:
(Will be provided in the observation)

Return ONLY raw JSON. Strip any markdown fences.""".format(
    red_flags=", ".join(RED_FLAG_VOCABULARY)
)


def build_single_task_prompt(obs: Observation, schema: str) -> str:
    """Build system + user message for single-patient task."""
    user_message = f"""
Patient Note:
{obs.note}

Available Wards: {", ".join(obs.available_wards)}
Bed Availability: {json.dumps(obs.bed_counts)}
Time of Day: {obs.time_of_day}

Output Format (TriageAction - raw JSON only):
{schema}

Provide ONLY the JSON output, no other text.
"""
    return user_message


def build_batch_task_prompt(obs: BatchObservation, schema: str) -> str:
    """Build system + user message for batch task with chain-of-thought guidance."""
    notes_str = "\n".join([f"[Patient {pid}] {note}" for pid, note in obs.notes.items()])
    patient_ids = list(obs.notes.keys())
    
    user_message = f"""
Patient Notes (5 patients):
{notes_str}

Available Wards: {", ".join(obs.available_wards)}
Total Beds Available: {json.dumps(obs.bed_counts)}
Time of Day: {obs.time_of_day}

PATIENT IDs TO PROCESS (ALL 5 MUST appear in priority_queue):
{", ".join(patient_ids)}

STEP 1 - Triage each patient individually:
For each patient (ID | Chief Complaint | ESI Level | Clinical Reasoning)

STEP 2 - Rank by urgency:
Create a list of all {len(patient_ids)} patient IDs ordered from MOST CRITICAL (rank 1) to LEAST (rank {len(patient_ids)})

STEP 3 - Allocate bed resources:
Available: 
  - 3 trauma capacity total
  - 2 ICU beds (critical)
  - Remaining wards as needed
Each patient gets exactly ONE ward assignment. No ward exceeds capacity.

STEP 4 - Output JSON:
Use this exact structure:
{schema}

CRITICAL CONSTRAINTS:
✓ priority_queue must contain ALL exactly {len(patient_ids)} patient IDs
✓ No duplicate patient IDs
✓ All patients in individual_triages
✓ No ward gets more patients than bed_count
✓ Respect clinical urgency in allocation

Provide ONLY the JSON output, no other text, no explanations.
"""
    return user_message


def test_inference(model: str, task_id: str, num_episodes: int = 3) -> Dict[str, Any]:
    """
    Test a model on a specific task.
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4", "claude-3-opus")
        task_id: Task identifier
        num_episodes: Number of test episodes
    
    Returns:
        Results dict with scores and metadata
    """
    
    env = MedTriageEnv()
    client = get_llm_client(model)
    
    episode_results = []
    
    # Log task start (hackathon format: [START] task=... env=... model=...)
    print(f"[START] task={task_id} env=medtriage model={model}")
    
    for episode_num in range(num_episodes):
        # Add throttling between episodes to reduce burst API calls (except first episode)
        if episode_num > 0:
            time.sleep(0.5)  # 500ms pause between requests
        
        try:
            obs = env.reset(task_id)
            start_time = time.time()
            
            # Call LLM API with OpenAI-compatible interface
            if isinstance(obs, Observation):
                user_prompt = build_single_task_prompt(obs, TriageAction.model_json_schema())
            else:
                user_prompt = build_batch_task_prompt(obs, BatchTriageAction.model_json_schema())
            
            # Retry with exponential backoff for rate limits
            max_retries = 2  # Reduced from 3 to save on retries
            response = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        max_tokens=1024,  # Reduced from 2000 (most responses ~300-600 tokens)
                        temperature=0.0,  # Deterministic responses for reproducibility
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    break  # Success, exit retry loop
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                        time.sleep(wait_time)
                    else:
                        # Graceful degradation: log error in hackathon format and continue with 0 reward
                        error_msg = "RateLimitError"
                        print(f"[STEP] step={episode_num + 1} action=llm_call reward=0.00 done=true error={error_msg}")
                        episode_results.append({
                            "episode": episode_num + 1,
                            "reward": 0.0,
                            "done": True,
                            "error": "RateLimitError"
                        })
                        continue  # Skip to next episode instead of crashing
            
            # Only process response if we got one successfully (skip on rate limit)
            if response is None:
                continue  # Skip to next episode
            
            # Extract JSON from response using robust parser
            response_text = response.choices[0].message.content
            action_data = robust_json_parse(response_text)
            
            if action_data is None:
                raise ValueError(f"Could not parse JSON from response: {response_text[:100]}...")
            
            # Clamp vital signs to valid ranges (handles model edge cases)
            action_data = clamp_vital_ranges(action_data)
            
            # Parse and validate with Pydantic
            if isinstance(obs, Observation):
                action = TriageAction(**action_data)
            else:
                action = BatchTriageAction(**action_data)
            
            elapsed = time.time() - start_time
            reward, done, info = env.step(action)
            
            episode_results.append({
                "episode": episode_num + 1,
                "reward": reward,
                "latency_ms": elapsed * 1000,
                "clarifications": info["clarifications"],
            })
            
            # Log step (hackathon format: [STEP] step=... action=... reward=... done=... error=...)
            done_str = "true" if done else "false"
            action_str = str(action)[:50]
            print(f"[STEP] step={episode_num + 1} action={action_str} reward={reward:.2f} done={done_str} error=null")
            
        except Exception as e:
            episode_results.append({
                "episode": episode_num + 1,
                "reward": 0.0,
                "error": str(e),
            })
            # Log step with error (hackathon format)
            error_msg = str(e)[:100]
            print(f"[STEP] step={episode_num + 1} action=llm_call reward=0.00 done=true error={error_msg}")
    
    # Aggregate results
    all_rewards = [r["reward"] for r in episode_results if "reward" in r]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    score = min(max(avg_reward, 0.0), 1.0)  # Clamp to [0, 1]
    success = score >= 0.5  # Success threshold
    
    # Log task end (hackathon format: [END] success=... steps=... score=... rewards=...)
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(f"[END] success={str(success).lower()} steps={num_episodes} score={score:.2f} rewards={rewards_str}")
    
    return {
        "model": model,
        "task_id": task_id,
        "num_episodes": num_episodes,
        "avg_reward": round(avg_reward, 4),
        "episodes": episode_results,
    }


def main():
    """CLI entry point or server launcher."""
    # Get default model from environment or fallback
    default_model = os.environ.get("MODEL_NAME", "gpt-4o")
    
    parser = argparse.ArgumentParser(description="MedTriage inference runner")
    parser.add_argument("--model", default=default_model, help=f"Model to test (default: {default_model})")
    parser.add_argument("--task", default="all", help="Task: task1, task2, task3, or all (default: all)")
    parser.add_argument("--output", default="results.json", help="Output file (default: results.json)")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task (default: 3)")
    parser.add_argument("--server", action="store_true", help="Run as Flask server instead of CLI")
    
    args = parser.parse_args()
    
    # Run Flask server if --server flag or RUNNING_ON_SPACES env var is set
    if args.server or os.environ.get("RUNNING_ON_SPACES"):
        port = int(os.environ.get("PORT", 7860))
        console.print(f"[bold green]Starting MedTriage OpenEnv server on port {port}...[/bold green]")
        app.run(host="0.0.0.0", port=port, debug=False)
        return
    
    # Otherwise run CLI mode
    tasks = {
        "task1": "task1_single_clear",
        "task2": "task2_ambiguous",
        "task3": "task3_batch_constrained",
    }
    
    if args.task == "all":
        task_ids = list(tasks.values())
    else:
        task_ids = [tasks.get(args.task, args.task)]
    
    console.print(Panel(f"[bold]MedTriage Benchmark - {args.model}[/bold]", style="blue"))
    console.print(f"\n[yellow]Note:[/yellow] Using environment variables:")
    if args.model.startswith("groq") or "mixtral" in args.model or "llama" in args.model or args.model.startswith("openai/"):
        console.print(f"  GROQ_API_KEY (get key at console.groq.com/keys)")
    else:
        console.print(f"  OPENAI_API_KEY (get key at platform.openai.com)")
    
    all_results = {
        "model": args.model,
        "timestamp": str(time.time()),
        "tasks": {}
    }
    
    for task_id in task_ids:
        result = test_inference(args.model, task_id, num_episodes=args.episodes)
        all_results["tasks"][task_id] = result
    
    # Print summary table
    table = Table(title=f"Results Summary - {args.model}")
    table.add_column("Task", style="cyan")
    table.add_column("Avg Reward", style="green")
    table.add_column("Episodes", style="magenta")
    
    for task_id, result in all_results["tasks"].items():
        table.add_row(
            task_id,
            f"{result['avg_reward']:.4f}",
            str(result["num_episodes"])
        )
    
    console.print(table)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    console.print(f"\n[bold green]✓ Results saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
