"""Download and verify a public Hugging Face dataset snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(
        f"https://huggingface.co/api/datasets/{args.repo_id}?blobs=true"
    )
    siblings = metadata.get("siblings", [])
    if not siblings:
        raise RuntimeError(f"No files found for dataset {args.repo_id}")

    downloaded = 0
    skipped = 0
    total_bytes = 0
    for item in siblings:
        filename = item["rfilename"]
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe dataset path: {filename}")

        destination = output_dir / relative_path
        lfs_info = item.get("lfs") or {}
        expected_size = lfs_info.get("size", item.get("size"))
        expected_sha256 = str(lfs_info.get("oid", "")).removeprefix("sha256:")

        if destination.exists():
            if expected_size is not None and destination.stat().st_size == expected_size:
                print(f"SKIP size_ok {filename}")
                skipped += 1
                total_bytes += destination.stat().st_size
                continue
            raise FileExistsError(f"Refusing to overwrite unexpected file: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        download_url = (
            f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/"
            f"{urllib.parse.quote(filename, safe='/')}?download=true"
        )
        actual_size, actual_sha256 = download_file(download_url, destination)

        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}"
            )
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        print(f"DOWNLOADED bytes={actual_size} sha256={actual_sha256} {filename}")
        downloaded += 1
        total_bytes += actual_size

    print(
        f"SNAPSHOT_OK repo={args.repo_id} files={len(siblings)} "
        f"downloaded={downloaded} skipped={skipped} bytes={total_bytes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "CabinAgent-RL/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_file(url: str, destination: Path) -> tuple[int, str]:
    temporary_path = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "CabinAgent-RL/1.0"})
    digest = hashlib.sha256()
    total_bytes = 0

    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as output_file:
            while chunk := response.read(1024 * 1024):
                output_file.write(chunk)
                digest.update(chunk)
                total_bytes += len(chunk)

    temporary_path.replace(destination)
    return total_bytes, digest.hexdigest()


if __name__ == "__main__":
    main()
