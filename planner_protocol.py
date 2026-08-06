import json
import os
import tempfile
import time
import traceback

from plan_history import record_attempt

MEMORY_RETRY_EXIT_CODE = 86
CANNOT_RUN_EXIT_CODE = 87
FATAL_EXIT_CODE = 88


def is_memory_error(error):
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    text = f"{type(error).__name__}: {error}".lower()
    phrases = [
        "out of memory",
        "cuda error: out of memory",
        "mps backend out of memory",
        "not enough memory",
        "cannot allocate memory",
        "defaultcpuallocator: can't allocate memory",
        "std::bad_alloc",
        "allocation failed",
    ]
    return any(phrase in text for phrase in phrases)


def atomic_json_write(path, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(prefix="planner_result_", suffix=".json", dir=directory)
    os.close(handle)
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary_path, path)


def result_entry(plan, result, workload=None, phase="", error=None):
    entry = {
        "timestamp": time.time(),
        "hardware_signature": plan.get("hardware_signature"),
        "model_key": plan.get("model_key"),
        "model_reference": plan.get("model_reference"),
        "plan_id": plan.get("plan_id"),
        "attempt_id": plan.get("attempt_id"),
        "result": result,
        "phase": phase,
        "workload": workload or {},
        "plan_summary": plan.get("status_text", ""),
    }

    if error is not None:
        entry["error_type"] = type(error).__name__
        entry["error"] = str(error)
        entry["traceback"] = traceback.format_exc()

    return entry


def report_result(plan, result, workload=None, phase="", error=None):
    entry = result_entry(plan, result, workload, phase, error)
    result_file = plan.get("result_file")
    if result_file:
        atomic_json_write(result_file, entry)
    record_attempt(entry)
    return entry


def report_success(plan, workload=None, phase="inference_complete"):
    if not plan or plan.get("success_reported"):
        return
    report_result(plan, "success", workload, phase)
    plan["success_reported"] = True


def report_memory_failure(plan, error, workload=None, phase="unknown"):
    report_result(plan, "memory_failure", workload, phase, error)


def report_fatal_failure(plan, error, workload=None, phase="unknown"):
    report_result(plan, "fatal_failure", workload, phase, error)


def exit_for_memory_retry(plan, error, workload=None, phase="unknown"):
    report_memory_failure(plan, error, workload, phase)
    os._exit(MEMORY_RETRY_EXIT_CODE)


def exit_for_fatal_failure(plan, error, workload=None, phase="unknown"):
    report_fatal_failure(plan, error, workload, phase)
    os._exit(FATAL_EXIT_CODE)
