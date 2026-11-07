#!/usr/bin/env python3
"""Catch accidentally staged models, builds, environments and oversized files."""
import pathlib
import subprocess
import sys
import re

paths = (
    subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    )
    .decode()
    .split("\0")
)
errors = []
for name in filter(None, paths):
    p = pathlib.PurePosixPath(name)
    if any(
        x in {"models", ".venv", "__pycache__", "node_modules"} or x.startswith("build")
        for x in p.parts
    ) or name.startswith("results/local/"):
        errors.append(name)
    if (
        p.name == ".env"
        or (p.name.startswith(".env.") and p.name != ".env.example")
        or p.suffix in {".gguf", ".safetensors", ".pyc", ".bin"}
    ):
        errors.append(name)
    # A submodule is a gitlink, not a blob: skip it deliberately.
    mode = (
        subprocess.check_output(["git", "ls-files", "--stage", "--", name])
        .decode()
        .split()[0]
    )
    if mode == "160000":
        continue
    data = subprocess.check_output(["git", "show", ":" + name])
    if len(data) > 2 * 1024 * 1024:
        errors.append(name + " (over 2 MiB)")
    if re.search(
        rb"(?m)^-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----\r?$",
        data,
    ):
        errors.append(name + " (private key)")
if errors:
    print(
        "Refusing accidental artifacts:\n" + "\n".join(sorted(set(errors))),
        file=sys.stderr,
    )
    sys.exit(1)
print(
    f"Reviewed {len(list(filter(None,paths)))} staged paths; no models/builds/local runs/environments/private keys."
)
