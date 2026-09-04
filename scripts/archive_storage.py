"""Copy approved project-relative artifacts to HDD and verify exact SHA-256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(f"Archive source is missing or unsafe: {path}")
    symlinks = [item for item in path.rglob("*") if item.is_symlink()] if path.is_dir() else []
    if symlinks:
        raise ValueError(f"Archive trees must not contain symlinks: {symlinks[0]}")
    items = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    base = path.parent if path.is_file() else path
    return [
        {
            "path": item.relative_to(base).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in items
    ]


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_relative(value: str) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or bool(relative.anchor)
        or value.startswith(("/", "\\"))
        or ".." in relative.parts
        or not relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"Archive item must be a safe project-relative path: {value}")
    return relative


def safe_batch_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe archive batch ID: {value}")
    return value


def ensure_archive_destination(archive_root: Path, relative: Path) -> Path:
    destination = archive_root / relative
    resolved = destination.resolve()
    if archive_root not in resolved.parents:
        raise ValueError(f"Archive destination escapes archive root: {relative}")
    current = destination.parent
    while current != archive_root:
        if current.is_symlink():
            raise ValueError(f"Archive destination parent is a symlink: {current}")
        current = current.parent
    return destination


def archive_items(
    source_root: Path,
    archive_root: Path,
    relatives: list[str],
    batch_id: str,
    soft_limit_bytes: int,
) -> dict[str, object]:
    source_root = source_root.resolve()
    archive_root = archive_root.resolve()
    batch_id = safe_batch_id(batch_id)
    relative_paths = [safe_relative(value) for value in relatives]
    if len({value.as_posix() for value in relative_paths}) != len(relative_paths):
        raise ValueError("Duplicate archive relative path")
    for left in relative_paths:
        for right in relative_paths:
            if left != right and left in right.parents:
                raise ValueError(f"Overlapping archive paths are not allowed: {left} and {right}")
    sources = [(source_root / value).resolve() for value in relative_paths]
    destinations = [ensure_archive_destination(archive_root, value) for value in relative_paths]
    for relative, source, destination in zip(relative_paths, sources, destinations, strict=True):
        if source_root not in source.parents and source != source_root:
            raise ValueError(f"Archive source escapes project root: {relative}")
        if not source.exists() or source.is_symlink():
            raise FileNotFoundError(f"Archive source is missing or unsafe: {source}")
        if source.is_dir() and any(item.is_symlink() for item in source.rglob("*")):
            raise ValueError(f"Archive source tree contains a symlink: {source}")
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")

    incoming_root = archive_root / ".incoming" / batch_id
    if incoming_root.exists():
        raise FileExistsError(f"Archive incoming batch already exists: {incoming_root}")
    source_bytes = sum(tree_size(source) if source.is_dir() else source.stat().st_size for source in sources)
    archive_bytes_before = tree_size(archive_root)
    if archive_bytes_before + source_bytes > soft_limit_bytes:
        raise RuntimeError(
            f"HDD soft limit would be exceeded: {archive_bytes_before + source_bytes} > {soft_limit_bytes}"
        )

    records: list[dict[str, object]] = []
    incoming_root.mkdir(parents=True, exist_ok=False)
    for relative, source in zip(relative_paths, sources, strict=True):
        temporary = incoming_root / relative
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, temporary)
        else:
            shutil.copy2(source, temporary)
        source_manifest = inventory(source)
        destination_manifest = inventory(temporary)
        if source_manifest != destination_manifest:
            raise RuntimeError(f"Archive verification failed for {relative}")
        records.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": sum(int(item["size_bytes"]) for item in source_manifest),
                "file_count": len(source_manifest),
                "files": source_manifest,
            }
        )

    for relative in relative_paths:
        temporary = incoming_root / relative
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    shutil.rmtree(incoming_root)
    return {
        "status": "verified",
        "batch_id": batch_id,
        "source_root": source_root.as_posix(),
        "archive_root": archive_root.as_posix(),
        "soft_limit_bytes": soft_limit_bytes,
        "archive_bytes_before": archive_bytes_before,
        "copied_bytes": source_bytes,
        "archive_bytes_after": tree_size(archive_root),
        "items": records,
        "source_deleted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--relative", action="append", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--soft-limit-gb", type=float, default=180.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = archive_items(
        Path(args.source_root),
        Path(args.archive_root),
        args.relative,
        args.batch_id,
        int(args.soft_limit_gb * 1_000_000_000),
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    print(report)


if __name__ == "__main__":
    main()
