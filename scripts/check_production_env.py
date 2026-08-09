#!/usr/bin/env python3
"""Check required production environment variables.

Used in CI to ensure deployment secrets like SERVICE_TOKEN_SECRET are present
via GitHub Actions secrets. Exits non-zero when any required var is missing.
"""
import os
import sys

REQUIRED = [
    "SERVICE_TOKEN_SECRET",
]

missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    print("Missing required production env vars:", ", ".join(missing))
    sys.exit(2)

print("All required production env vars present.")
