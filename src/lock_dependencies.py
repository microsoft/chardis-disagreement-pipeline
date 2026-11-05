"""Regenerate pip-compatible platform locks with the separately installed uv tool."""

import argparse
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"macos-arm64": "aarch64-apple-darwin", "linux-x86_64": "x86_64-manylinux_2_28"}


def normalize_hashes(path):
    text = path.read_text()
    # Some mirrors advertise MD5 alongside SHA256. pip accepts only strong hashes.
    text = re.sub(r"^ +--hash=md5:[a-f0-9]+(?: \\)?\n", "", text, flags=re.M)
    for requirement in re.split(r"\n(?=[a-zA-Z0-9])", text):
        if "==" in requirement and "--hash=sha256:" not in requirement:
            raise ValueError(f"Lock contains a requirement without SHA256 in {path.name}.")
    if "--hash=md5:" in text:
        raise ValueError("Unsupported hash format in resolver output.")
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, action="append")
    args = parser.parse_args()
    local_resolver = Path(sys.executable).parent / "uv"
    resolver = str(local_resolver) if local_resolver.is_file() else shutil.which("uv")
    if not resolver:
        raise RuntimeError("Install requirements-tools.txt before regenerating locks.")
    index = os.environ.get("PIP_INDEX_URL")
    if not index:
        output = subprocess.check_output([sys.executable, "-m", "pip", "config", "list"], text=True)
        settings = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if "global.index-url" in settings:
            index = ast.literal_eval(settings["global.index-url"])
    (ROOT / "locks").mkdir(parents=True, exist_ok=True)
    for target in args.target or TARGETS:
        environment = os.environ.copy()
        if target == "macos-arm64":
            environment["MACOSX_DEPLOYMENT_TARGET"] = "14.0"
        for profile in ("", "browser"):
            suffix = "-" + profile if profile else ""
            path = ROOT / "locks" / f"py311-{target}{suffix}.txt"
            command = [resolver, "pip", "compile", f"requirements{suffix}.txt",
                       "--python-version", "3.11", "--python-platform", TARGETS[target],
                       "--generate-hashes", "--no-header", "-o", str(path)]
            if profile:
                command.extend(["-c", str(ROOT / "locks" / f"py311-{target}.txt")])
            if index:
                command.extend(["--index-url", index])
            try:
                subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, env=environment)
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Dependency resolution failed for {path.name}.") from None
            normalize_hashes(path)
            print(f"Generated {path.name}")


if __name__ == "__main__":
    main()
