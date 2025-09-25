#!/usr/bin/env python3

"""
Lightweight smoke-check for critical imports so users can quickly identify missing
optional dependencies without running full apps. Exits nonzero on failure.
"""

import importlib
import sys

REQUIRED_IMPORTS = [
    "torch",
    "torchaudio",
    "librosa",
    "yaml",
    "gradio",
]

OPTIONAL_IMPORTS = [
    "runpod",  # serverless
    "pyaudio",  # recording
]


def check(names, optional=False):
    ok = True
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:
            level = "WARN" if optional else "ERROR"
            print(f"[{level}] import failed: {name}: {e}")
            if not optional:
                ok = False
    return ok


def main():
    req_ok = check(REQUIRED_IMPORTS, optional=False)
    _ = check(OPTIONAL_IMPORTS, optional=True)
    if not req_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()


