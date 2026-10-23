"""Build reviewable source, portable and third-party-source release archives.

The source allowlist deliberately excludes environments, captures, profiles,
generated benchmarks, VCS history and the parent workspace. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from fetch_license_sources import file_matches, validate_manifest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("subtitle_eraser", "tools", "tests", "docs", "licenses", ".github")
SOURCE_FILES = ("README.md", "LICENSE", "THIRD_PARTY.md", "CONTRIBUTING.md", "CHANGELOG.md",
                "pyproject.toml", "MANIFEST.in", ".gitignore", ".gitattributes", "SubtitleEraser.spec",
                "requirements.txt", "requirements.lock.txt", "requirements-dev.txt", "research-requirements.txt",
                "setup.cmd", "start.cmd", "build.cmd", "validate.cmd", "setup-research.cmd", "research.cmd")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv"}


def source_files(root=ROOT):
    selected = [root / name for name in SOURCE_FILES]
    for name in SOURCE_DIRS:
        selected.extend(path for path in (root / name).rglob("*") if path.is_file()
                        and not SKIP_PARTS.intersection(path.relative_to(root).parts)
                        and path.suffix.lower() not in {".pyc", ".pyo", ".log"})
    for path in selected:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe source path: {path.name}")
    return sorted(set(selected), key=lambda path: path.relative_to(root).as_posix())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def make_archives(root, dist, output, sources):
    version = re.search(r'__version__ = "([0-9.]+)"', (root / "subtitle_eraser/__init__.py").read_text()).group(1)
    name = f"SubtitleEraser-{version}"
    files = source_files(root)
    if not (dist / "SubtitleEraser.exe").is_file() or not (dist / "_internal").is_dir():
        raise ValueError("Build and test the complete portable directory first")
    for required in ("LICENSE", "THIRD_PARTY.md", "licenses/source-manifest.json"):
        if not (dist / required).is_file():
            raise ValueError(f"Portable documentation missing: {required}; run package_reports.py")
    manifest = validate_manifest(json.loads((root / "licenses/source-manifest.json").read_text(encoding="utf-8")))
    for entry in manifest:
        if not file_matches(sources / entry["file"], entry):
            raise ValueError(f"Verified third-party source missing: {entry['file']}; run fetch_license_sources.py")
    output.mkdir(parents=True, exist_ok=True)
    source_zip, windows_zip, third_zip = (output / f"{name}-{suffix}.zip" for suffix in
                                         ("Source", "Windows-x64", "ThirdPartySources"))
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f"{name}/{path.relative_to(root).as_posix()}")
    with zipfile.ZipFile(windows_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(dist.rglob("*")):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(dist.resolve()):
                    raise ValueError("Portable directory contains an unsafe link")
                archive.write(path, f"SubtitleEraser/{path.relative_to(dist).as_posix()}")
    with zipfile.ZipFile(third_zip, "w", zipfile.ZIP_STORED) as archive:
        for entry in manifest:
            archive.write(sources / entry["file"], entry["file"])
        archive.write(root / "licenses/source-manifest.json", "source-manifest.json")
        archive.write(root / "THIRD_PARTY.md", "README.md")
        for path in sorted((root / "licenses").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    archives = [source_zip, windows_zip, third_zip]
    checksums = output / "SHA256SUMS.txt"
    checksums.write_text("".join(f"{digest(path)}  {path.name}\n" for path in archives), encoding="ascii")
    kit = output / f"{name}-GitHub-ReleaseKit.zip"
    with zipfile.ZipFile(kit, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f"project/{path.relative_to(root).as_posix()}")
        for path in [*archives, checksums]:
            archive.write(path, f"release-assets/{path.name}", compress_type=zipfile.ZIP_STORED)
        archive.write(root / "docs/PUBLISHING.md", "先读我-发布指南.md")
    # Read every compressed member, catching truncated files before handoff.
    for path in [*archives, kit]:
        with zipfile.ZipFile(path) as archive:
            failure = archive.testzip()
            if failure:
                raise RuntimeError(f"Corrupt archive member: {failure}")
    (output / "release-manifest.json").write_text(json.dumps({
        "version": version, "source_files": len(files),
        "artifacts": [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                      for path in [*archives, kit]],
        "publication": "Prepared locally; no GitHub upload performed"
    }, indent=2), encoding="utf-8")
    return kit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", type=Path, default=ROOT / "dist-release")
    parser.add_argument("--output", type=Path, default=ROOT / "release")
    parser.add_argument("--sources", type=Path, default=ROOT / ".release-cache/third-party-sources")
    args = parser.parse_args()
    print(make_archives(ROOT, args.dist_root.resolve() / "SubtitleEraser", args.output.resolve(), args.sources.resolve()))


if __name__ == "__main__":
    main()
