"""Fail when Radon reports high cyclomatic complexity in the core application codebase."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

TARGET_PATHS = ("Dashboard", "tests")
BAD_GRADE_PATTERN = re.compile(r" - [D-F] \(")
GRADE_PATTERN = re.compile(r" - ([A-F]) \(")


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "-s", *TARGET_PATHS],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    report = result.stdout
    counts = Counter(GRADE_PATTERN.findall(report))

    print("Cyclomatic complexity grade counts for application code:")
    for grade in "ABCDEF":
        print(f"  {grade}: {counts.get(grade, 0)}")

    bad_lines = [line for line in report.splitlines() if BAD_GRADE_PATTERN.search(line)]
    if bad_lines:
        print("\nD-or-worse complexity blocks found in application code:")
        print("\n".join(bad_lines))
        return 1

    print("\n✅ All application modules within cyclomatic complexity limits (Grades A-C).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
