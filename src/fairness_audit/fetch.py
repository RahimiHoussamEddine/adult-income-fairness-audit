"""Download and checksum the official UCI archive."""

from __future__ import annotations

import hashlib
import urllib.request

from fairness_audit.constants import RAW_DIR, SOURCE_BYTES, SOURCE_SHA256, SOURCE_URL


def fetch_source() -> dict[str, str | int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "adult-income-fairness-audit/0.1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise ValueError(
            "UCI archive changed: expected "
            f"{SOURCE_BYTES} bytes and SHA-256 {SOURCE_SHA256}; "
            f"received {len(payload)} bytes and {digest}"
        )
    target = RAW_DIR / "adult.zip"
    target.write_bytes(payload)
    return {"path": str(target), "bytes": len(payload), "sha256": digest}


if __name__ == "__main__":
    print(fetch_source())
