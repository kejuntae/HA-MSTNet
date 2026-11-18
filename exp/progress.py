"""Report live progress without mixing it into JSON output."""

from __future__ import annotations

from datetime import datetime
import sys


def print_progress(stage: str, message: str) -> None:
    """Write timestamped progress to stderr immediately."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [{stage}] {message}",
        file=sys.stderr,
        flush=True,
    )
