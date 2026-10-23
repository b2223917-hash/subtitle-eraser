"""Fetch the exact third-party source archives recorded for a release.

The manifest pins HTTPS URLs, sizes and SHA-256 hashes. This tool never executes
or extracts downloaded content and needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 1024 * 1024


def validate_manifest(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Source manifest must be a non-empty array")
    entries, filenames = [], set()
    for entry in value:
        if not isinstance(entry, dict) or not {"file", "url", "sha256", "bytes"} <= entry.keys():
            raise ValueError("Every source needs file, url, sha256 and bytes")
        filename = entry["file"]
        if (not isinstance(filename, str) or not filename or filename in (".", "..") or
                any(character in filename for character in '/\\:<>"|?*') or
                any(ord(character) < 32 for character in filename) or
                filename.endswith((".", " "))):
            raise ValueError("Source filenames must be plain portable filenames")
        if filename.casefold() in filenames:
            raise ValueError(f"Duplicate source filename: {filename}")
        filenames.add(filename.casefold())
        address = urlsplit(entry["url"]) if isinstance(entry["url"], str) else None
        if (address is None or address.scheme != "https" or not address.hostname or
                address.username is not None or address.password is not None):
            raise ValueError(f"Source URL must use HTTPS without credentials: {filename}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {filename}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Invalid source size for {filename}")
        entries.append({"file": filename, "url": entry["url"],
                        "sha256": entry["sha256"].lower(), "bytes": entry["bytes"]})
    return entries


def file_matches(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def fetch_source(entry: dict, destination: Path, *, attempts=3, timeout=60) -> Path:
    entry = validate_manifest([entry])[0]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / entry["file"]
    if file_matches(target, entry):
        print(f"Verified cached source: {entry['file']}", flush=True)
        return target
    for attempt in range(attempts):
        temporary = None
        try:
            print(f"Downloading source ({attempt + 1}/{attempts}): {entry['file']}", flush=True)
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "SubtitleEraser-source-fetch/1"})
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Source download redirected away from HTTPS")
                with tempfile.NamedTemporaryFile(mode="wb", dir=destination, prefix=".source-",
                                                 suffix=".part", delete=False) as handle:
                    temporary = Path(handle.name)
                    while chunk := response.read(CHUNK_SIZE):
                        size += len(chunk)
                        if size > entry["bytes"]:
                            raise ValueError(f"Source exceeds expected size: {entry['file']}")
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Source size or SHA-256 mismatch: {entry['file']}")
            os.replace(temporary, target)
            print(f"Verified downloaded source: {entry['file']}", flush=True)
            return target
        except (OSError, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Could not fetch verified source {entry['file']}: {error}") from error
            time.sleep(min(2 ** attempt, 4))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError("attempts must be positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "licenses" / "source-manifest.json")
    parser.add_argument("--destination", type=Path, default=ROOT / ".release-cache" / "third-party-sources")
    args = parser.parse_args(argv)
    entries = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    for entry in entries:
        fetch_source(entry, args.destination)
    print(f"Verified {len(entries)} source archives in {args.destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
