import os
import py_compile
import sys
import unittest
from pathlib import Path

from model_registry import MODEL_PROFILES, VALIDATION_ERRORS


def main():
    base_dir = Path(__file__).resolve().parent
    source_files = list(base_dir.glob("*.py")) + list((base_dir / "tests").glob("*.py"))
    try:
        for source_file in source_files:
            py_compile.compile(source_file, doraise=True)
    except py_compile.PyCompileError as error:
        print(error)
        print("Python compilation failed.")
        return 1

    if VALIDATION_ERRORS:
        print("Model registry validation failed:")
        for error in VALIDATION_ERRORS:
            print(f"- {error}")
        return 1

    suite = unittest.defaultTestLoader.discover(os.path.join(base_dir, "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    print(f"\nValidated {len(MODEL_PROFILES)} model profiles and the planner test suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
