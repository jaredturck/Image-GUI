import compileall
import os
import sys
import unittest

from model_registry import MODEL_PROFILES, VALIDATION_ERRORS


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compiled = compileall.compile_dir(base_dir, quiet=1)
    if not compiled:
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
