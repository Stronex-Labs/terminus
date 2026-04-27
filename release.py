#!/usr/bin/env python3
"""Release script: bump version, build, upload to PyPI, commit & push.

Usage:
    python release.py patch    # 0.1.2 -> 0.1.3
    python release.py minor    # 0.1.2 -> 0.2.0
    python release.py major    # 0.1.2 -> 1.0.0
    python release.py 0.2.1    # explicit version
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOML = Path("pyproject.toml")


def current_version():
    m = re.search(r'^version\s*=\s*"([^"]+)"', TOML.read_text(), re.MULTILINE)
    if not m:
        sys.exit("Could not find version in pyproject.toml")
    return m.group(1)


def bump(ver: str, part: str) -> str:
    major, minor, patch = (int(x) for x in ver.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return part  # explicit version string


def run(*cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    old = current_version()
    new = bump(old, part)

    print(f"\nReleasing terminus-lab  {old} -> {new}\n")

    # 1. Write new version
    text = TOML.read_text()
    text = re.sub(r'^(version\s*=\s*)"[^"]+"', f'\\1"{new}"', text, flags=re.MULTILINE)
    TOML.write_text(text)
    print(f"[1/5] Bumped pyproject.toml to {new}")

    # 2. Commit & tag
    run("git", "add", "pyproject.toml")
    run("git", "commit", "-m", f"chore: release {new}")
    run("git", "tag", f"v{new}")
    print(f"[2/5] Committed and tagged v{new}")

    # 3. Push
    run("git", "push", "origin", "main")
    run("git", "push", "origin", f"v{new}")
    print("[3/5] Pushed to GitHub")

    # 4. Build
    if Path("dist").exists():
        shutil.rmtree("dist")
    run(sys.executable, "-m", "build")
    print("[4/5] Built distribution")

    # 5. Upload
    run(sys.executable, "-m", "twine", "upload", "dist/*")
    print(f"[5/5] Uploaded to PyPI\n")
    print(f"Done. terminus-lab {new} is live at https://pypi.org/project/terminus-lab/{new}/")


if __name__ == "__main__":
    main()
