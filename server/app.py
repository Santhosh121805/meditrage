"""
MedTriage OpenEnv Flask Server - HTTP API for inference
"""

import json
import os
import sys
import time
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import after env setup
from openai import OpenAI, RateLimitError
from models.schemas import TriageAction, BatchTriageAction, Observation, BatchObservation
from medtriage_env import MedTriageEnv
from utils.json_repair import robust_json_parse, clamp_vital_ranges

# Create Flask app
app = Flask(__name__)

# Global state for OpenEnv API
_env = None
_client = None
_current_observation = None
_model_name = None
_task_id = None


def get_llm_client(model: str) -> OpenAI:
    """Initialize OpenAI-compatible client using HF_TOKEN and API_BASE_URL."""
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


def _get_or_init_env_client(model: str):
    """Initialize environment and client if not already done."""
    global _env, _client, _model_name
    if _env is None:
        _env = MedTriageEnv()
    if _client is None or model != _model_name:
        _client = get_llm_client(model)
        _model_name = model
    return _env, _client


# ============ API Endpoints ============

@app.route("/", methods=["GET"])
def root_endpoint():
    """Root endpoint - confirms server is running."""
    return jsonify({"message": "MedTriage OpenEnv Server", "status": "running"}), 200


@app.route("/health", methods=["GET"])
def health_endpoint():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


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
    
    Accepts task_id from:
    1. URL path: POST /reset/task1_single_clear
    2. Query parameter: POST /reset?task_id=task1_single_clear
    3. JSON body: POST /reset with {"task_id": "task1_single_clear"}
    """
    try:
        print(f"[DEBUG] reset_endpoint called with path task_id={task_id}", file=sys.stderr)
        
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
            print(f"[DEBUG] No task_id found, using default: task1_single_clear", file=sys.stderr)
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
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


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
        global _current_observation, _env, _client
        
        print(f"[DEBUG] /step endpoint called", file=sys.stderr)
        
        if _env is None or _current_observation is None:
            print(f"[DEBUG] Environment not initialized", file=sys.stderr)
            return jsonify({"error": "environment not initialized, call /reset first"}), 400
        
        print(f"[DEBUG] Current observation type: {type(_current_observation).__name__}", file=sys.stderr)
        
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


def run_server(port: int = 7860, debug: bool = False):
    """Run the Flask server."""
    print(f"Starting MedTriage OpenEnv server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server()
