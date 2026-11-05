"""Verify the installed distributions against the selected hashed pip lock."""

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def lock_path(profile=""):
    if sys.version_info[:2] != (3, 11) or platform.python_implementation() != "CPython":
        raise RuntimeError("Use CPython 3.11 for the supported environment.")
    target = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64"}.get(
        (platform.system(), platform.machine()))
    if target is None:
        raise RuntimeError("This platform is outside the verified runtime matrix.")
    if profile not in {"", "browser"}:
        raise ValueError("Unknown environment profile.")
    suffix = "-" + profile if profile else ""
    return ROOT / "locks" / f"py311-{target}{suffix}.txt"


def verify():
    path = lock_path()
    errors = []
    paths = [path]
    for package, profile in (("playwright", "browser"),):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        paths.append(lock_path(profile))
    for selected in paths:
        text = selected.read_text()
        requirements = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", text, re.M)
        if not requirements or "--hash=sha256:" not in text:
            raise RuntimeError(f"Missing hashed requirements in {selected.name}.")
        for name, version in requirements:
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                errors.append(name)
    if errors:
        raise RuntimeError("Environment differs from lock. Run explicit --setup. Packages: " + ", ".join(errors))
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--profile", default="", choices=["", "browser"])
    args = parser.parse_args()
    if args.lock:
        print(lock_path(args.profile))
    else:
        verify()
