from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


class PipelineLogger:
    def __init__(self, log_dir: str | Path, run_name: str = "pipeline"):
        self.log_dir = ensure_dir(log_dir)
        self.run_name = run_name
        self.main_log_path = self.log_dir / "pipeline.log"
        self.events_path = self.log_dir / "pipeline_events.jsonl"

    def event(self, stage: str, message: str, **fields: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "run_name": self.run_name,
            "stage": stage,
            "message": message,
            **fields,
        }
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"[{timestamp}] [{stage}] {message}"
        if details:
            line = f"{line} {details}"
        print(line, flush=True)
        with self.main_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
