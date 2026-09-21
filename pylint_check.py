"""Script to execute pylint code quality analysis on the modules directory."""

import glob
import os
from pylint.lint import Run


def run_pylint_check() -> float:
    """Run pylint on all Python files inside modules/ directory.

    Returns:
        float: Average clean code score out of 10.
    """
    module_files = glob.glob(os.path.join("modules", "*.py"))
    print("=" * 60)
    print("RUNNING PYLINT CLEAN CODE AUDIT ON MODULES...")
    print("=" * 60)
    print(f"Target files: {module_files}\n")

    # Pass disable import-error and duplicate-code flags for static code analysis
    args = module_files + ["--disable=import-error,duplicate-code"]
    results = Run(args, exit=False)
    score = results.linter.stats.global_note

    print("\n" + "=" * 60)
    print(f"FINAL PYLINT CLEAN CODE SCORE: {score:.2f} / 10.00")
    print("=" * 60)

    if score >= 8.5:
        print("SUCCESS: Code quality satisfies Dicoding 5-Star Clean Code standards!")
    else:
        print("WARNING: Score is below 8.5/10. Review refactoring recommendations.")

    return score


if __name__ == "__main__":
    run_pylint_check()
