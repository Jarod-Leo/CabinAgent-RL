import unittest
from pathlib import Path

from scripts.download_models import download_with_retry, retry_delay


class _Response:
    status_code = 429
    headers = {"retry-after": "153"}


class _RateLimitError(Exception):
    response = _Response()


class DownloadModelTests(unittest.TestCase):
    def test_retry_after_header_is_honored(self) -> None:
        self.assertEqual(retry_delay(_RateLimitError(), 1, 30), 153)

    def test_transient_failure_retries_without_touching_target(self) -> None:
        calls = []
        sleeps = []

        def snapshot_fn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _RateLimitError("limited")
            return "/models/Qwen/test"

        result = download_with_retry(
            "Qwen/test",
            Path("unused"),
            snapshot_fn,
            max_attempts=2,
            base_delay=30,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(result, "/models/Qwen/test")
        self.assertEqual(sleeps, [153])
        self.assertEqual(calls[0]["max_workers"], 4)


if __name__ == "__main__":
    unittest.main()
