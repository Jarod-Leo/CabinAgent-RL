"""Validate downloaded CAR-bench and BFCL benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    car_root = Path(args.car_root)
    bfcl_root = Path(args.bfcl_root)

    car_task_files = sorted((car_root / "tasks").glob("*.jsonl"))
    bfcl_files = sorted(bfcl_root.rglob("BFCL_v4*.json"))
    car_mock_files = sorted((car_root / "mock_data").rglob("*.jsonl"))

    if len(car_task_files) != 6:
        raise RuntimeError(f"Expected 6 CAR task files, found {len(car_task_files)}")
    if not bfcl_files:
        raise RuntimeError("No BFCL V4 data files found")
    if not car_mock_files:
        raise RuntimeError("No CAR mock-data files found")

    car_records = sum(validate_json_records(path) for path in car_task_files)
    bfcl_records = sum(validate_json_records(path) for path in bfcl_files)
    for path in car_mock_files:
        validate_first_record(path)

    print(
        "OFFICIAL_DATA_OK "
        f"car_task_files={len(car_task_files)} car_task_records={car_records} "
        f"car_mock_files={len(car_mock_files)} bfcl_files={len(bfcl_files)} "
        f"bfcl_records={bfcl_records}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car-root", required=True)
    parser.add_argument("--bfcl-root", required=True)
    return parser.parse_args()


def validate_json_records(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    else:
        records = document if isinstance(document, list) else [document]

    if not records:
        raise RuntimeError(f"No records found in {path}")

    first_record = records[0]
    first_id = None
    if isinstance(first_record, dict):
        first_id = first_record.get("id", first_record.get("task_id"))
    print(f"JSON_OK records={len(records)} first_id={first_id} path={path}")
    return len(records)


def validate_first_record(path: Path) -> None:
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                json.loads(line)
                print(f"MOCK_OK path={path}")
                return
    raise RuntimeError(f"No records found in {path}")


if __name__ == "__main__":
    main()
