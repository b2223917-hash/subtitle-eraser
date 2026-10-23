"""Package documentation and runtime licenses; benchmark reports are explicit opt-in."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path, PurePosixPath
import re
import shutil


ROOT = Path(__file__).resolve().parent.parent


def runtime_distributions(root: Path) -> list[str]:
    names = []
    for line in (root / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==[^\s;]+", line.strip())
        if match:
            names.append(match.group(1))
    if not names:
        raise RuntimeError("No pinned runtime dependencies found in requirements.lock.txt")
    return names


def copy_dependency_licenses(root: Path, destination: Path) -> None:
    # Only runtime lock entries are distributed. Test/build dependencies, local
    # research environments and unrelated globally installed packages are excluded.
    for name in runtime_distributions(root):
        distribution = importlib.metadata.distribution(name)
        for file in distribution.files or []:
            if not any(token in str(file).lower() for token in ("license", "copying", "copyright", "notice")):
                continue
            source = Path(distribution.locate_file(file))
            if not source.is_file():
                continue
            # Distribution metadata may contain ../ paths. Preserve grouping
            # while preventing any copied name from escaping the license folder.
            parts = [part if part not in ("..", ".") else "_"
                     for part in PurePosixPath(str(file).replace("\\", "/")).parts]
            if not parts or any(":" in part or part.startswith("/") for part in parts):
                continue
            target = destination / "licenses" / name / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_optional_reports(root: Path, destination: Path) -> None:
    report = root / "outputs" / "validation"
    if report.is_dir():
        target = destination / "outputs" / "validation"
        target.mkdir(parents=True, exist_ok=True)
        for source in report.iterdir():
            if source.is_file() and source.suffix in (".html", ".json", ".mp4", ".gif", ".png"):
                shutil.copy2(source, target / source.name)
        for source in report.glob("propainter/*/comparison.mp4"):
            target_file = target / source.relative_to(report)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_file)
    for name in ("low-latency", "transition", "calibration"):
        source = root / "outputs" / name
        if source.is_dir():
            shutil.copytree(source, destination / "outputs" / name, dirs_exist_ok=True)


def package(root: Path, destination: Path, *, include_reports: bool = False) -> None:
    if not (destination / "SubtitleEraser.exe").is_file():
        raise RuntimeError("Build the portable app first")
    # A source archive/checkout need not contain any generated reports.
    for name in ("README.md", "THIRD_PARTY.md", "LICENSE"):
        shutil.copy2(root / name, destination / name)
    for name in ("licenses", "docs"):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)
    copy_dependency_licenses(root, destination)
    if include_reports:
        copy_optional_reports(root, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", default="dist-calibrated")
    parser.add_argument("--include-reports", action="store_true",
                        help="Copy local generated reports as well; review them before distribution")
    args = parser.parse_args()
    destination = ROOT / args.dist_root / "SubtitleEraser"
    package(ROOT, destination, include_reports=args.include_reports)
    print(f"Portable app and documentation ready: {destination}")


if __name__ == "__main__":
    main()
