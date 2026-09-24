import json
import time
from pathlib import Path
from typing import Any


class JSONLLogger:
    """Append-only JSONL pro Run.

    Layout:
        <log_dir>/<config_hash>/
            config.json     # Snapshot der PipelineConfig
            metrics.jsonl   # eine Zeile pro logged event
            result.json     # finales Ergebnis (best metrics + Konfig-Hash)
    """

    def __init__(self, log_dir: Path | str, config_hash: str, config_dict: dict):
        self.run_dir = Path(log_dir) / config_hash
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.result_path = self.run_dir / "result.json"

        self.config_path.write_text(json.dumps(config_dict, indent=2, default=str))
        self._t0 = time.time()
        self._fp = self.metrics_path.open("a", buffering=1)

    def log(self, **fields: Any) -> None:
        record = {"t_s": round(time.time() - self._t0, 3), **fields}
        self._fp.write(json.dumps(record, default=str) + "\n")

    def write_result(self, **fields: Any) -> None:
        payload = {"t_total_s": round(time.time() - self._t0, 3), **fields}
        self.result_path.write_text(json.dumps(payload, indent=2, default=str))

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
