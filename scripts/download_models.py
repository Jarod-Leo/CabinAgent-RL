"""Download immutable local snapshots for the policy and simulator models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable


MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
)


def retry_delay(error: BaseException, attempt: int, base_delay: float) -> float | None:
    """Return a bounded retry delay for transient Hub/network failures."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
            retry_after = getattr(response, "headers", {}).get("retry-after")
            try:
                return min(max(float(retry_after), base_delay), 600.0)
            except (TypeError, ValueError):
                return min(base_delay * (2 ** (attempt - 1)), 600.0)
        if type(current).__name__ in {
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "TimeoutError",
        }:
            return min(base_delay * (2 ** (attempt - 1)), 600.0)
        current = current.__cause__ or current.__context__
    return None


def download_with_retry(
    repo_id: str,
    target: Path,
    snapshot_fn: Callable[..., str],
    *,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return snapshot_fn(repo_id=repo_id, local_dir=target, max_workers=4)
        except Exception as error:
            delay = retry_delay(error, attempt, base_delay)
            if delay is None or attempt == max_attempts:
                raise
            print(
                f"MODEL_DOWNLOAD_RETRY repo={repo_id} attempt={attempt}/{max_attempts} "
                f"delay_seconds={delay:.0f} error={type(error).__name__}",
                flush=True,
            )
            sleep_fn(delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--base-delay", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.base_delay <= 0:
        parser.error("--max-attempts and --base-delay must be positive")

    from huggingface_hub import snapshot_download

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for repo_id in MODELS:
        target = output_root / repo_id
        target.mkdir(parents=True, exist_ok=True)
        path = download_with_retry(
            repo_id,
            target,
            snapshot_download,
            max_attempts=args.max_attempts,
            base_delay=args.base_delay,
        )
        manifest.append({"repo_id": repo_id, "path": path})
        print(f"MODEL_READY repo={repo_id} path={path}")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
