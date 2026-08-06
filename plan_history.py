import json
import os
import tempfile
import threading
import time

from app_config import config_dir

HISTORY_FILE = os.path.join(config_dir(), "plan_history.json")
HISTORY_LOCK = threading.Lock()


def empty_history():
    return {"schema_version": 1, "attempts": []}


def load_history():
    if not os.path.isfile(HISTORY_FILE):
        return empty_history()

    with HISTORY_LOCK:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                history = json.load(file)
        except (OSError, json.JSONDecodeError):
            return empty_history()
    if not isinstance(history, dict):
        return empty_history()
    history.setdefault("attempts", [])
    return history


def save_history(history):
    os.makedirs(config_dir(), exist_ok=True)
    with HISTORY_LOCK:
        handle, temporary_path = tempfile.mkstemp(prefix="plan_history_", suffix=".json", dir=config_dir())
        os.close(handle)
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)
        os.replace(temporary_path, HISTORY_FILE)


def record_attempt(entry):
    history = load_history()
    stored = dict(entry)
    stored.setdefault("timestamp", time.time())
    history.setdefault("attempts", []).append(stored)
    history["attempts"] = history["attempts"][-2000:]
    save_history(history)


def numeric_workload(workload):
    return {
        key: float(value)
        for key, value in (workload or {}).items()
        if isinstance(value, (int, float))
    }


def workload_covers(validated, requested):
    validated_values = numeric_workload(validated)
    requested_values = numeric_workload(requested)

    for key, requested_value in requested_values.items():
        if key not in validated_values:
            continue
        if requested_value > validated_values[key]:
            return False
    return True


def successful_plan(hardware_signature, model_key, workload):
    history = load_history()
    for attempt in reversed(history.get("attempts", [])):
        if attempt.get("result") != "success":
            continue
        if attempt.get("hardware_signature") != hardware_signature:
            continue
        if attempt.get("model_key") != model_key:
            continue
        if workload_covers(attempt.get("workload", {}), workload):
            return attempt
    return None


def failed_attempt_ids(hardware_signature, model_key, workload):
    history = load_history()
    failed = set()
    for attempt in history.get("attempts", []):
        if attempt.get("result") != "memory_failure":
            continue
        if attempt.get("hardware_signature") != hardware_signature:
            continue
        if attempt.get("model_key") != model_key:
            continue
        if workload_covers(workload, attempt.get("workload", {})):
            failed.add(attempt.get("attempt_id"))
    return failed
