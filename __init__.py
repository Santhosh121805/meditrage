"""MedTriage - Clinical ED Triage Benchmark"""

__version__ = "1.0.0"
__author__ = "MedTriage Team"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "MedTriageEnv":
        from medtriage_env import MedTriageEnv
        return MedTriageEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MedTriageEnv"]
