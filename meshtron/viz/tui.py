"""
tui.py

Simple terminal UI for the unified pipeline (train.py / Trainer / polytron_chain.py):
pick model family + dimension, edit the PipelineConfig fields that matter day to
day, pick a dataset file, save/load the config as JSON, and start training with
a live log pane -- without hand-building --flags or a JSON file first.

Deliberately NOT a general-purpose settings editor for every PipelineConfig
field (there are ~35): exposes the ones someone actually changes between runs,
grouped the same way config.py itself groups them. Anything else is still
reachable by loading a JSON config built elsewhere (train.py --config still
works standalone) or editing after "Save Config".

Run:
    uv run textual run tui.py     # or: python tui.py
"""

import json
import queue
import sys
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from meshtron.training.config import PipelineConfig


def _discover_datasets() -> list[str]:
    """*.pt files in data/, most-recently-modified first -- same convention
    the README documents."""
    here = Path("data")
    files = sorted(here.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


class _QueueWriter:
    """Redirect target for stdout/stderr during a training run -- captures
    both prints (Trainer's epoch summaries, polytron_chain's stage banners)
    and tqdm's progress bars (which write to stderr by default) into the same
    thread-safe queue the UI polls from, without threading a callback through
    every training function individually."""

    def __init__(self, q: "queue.Queue[str]"):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        # tqdm updates a line in place with \r; forward each \r/\n-terminated
        # chunk as one queue item so the log pane shows the latest progress
        # rather than every intermediate character.
        self._buf += s
        while "\r" in self._buf or "\n" in self._buf:
            for sep in ("\r", "\n"):
                if sep in self._buf:
                    line, self._buf = self._buf.split(sep, 1)
                    if line.strip():
                        self.q.put(line)
                    break
        return len(s)

    def flush(self) -> None:
        pass


FIELD_SPECS = {
    # (section, label, cfg attr, widget kind, kwargs)
    "pipeline": [
        ("model_family", "Model family", "select", {"options": [("Quadtron", "quadtron"), ("Polytron", "polytron")]}),
        ("dim", "Dimension", "select", {"options": [("2D", "2"), ("3D", "3")]}),
    ],
    "data": [
        ("data_path", "Data path", "input", {}),
        ("train_val_ratio", "Train/val ratio", "input", {}),
        ("n_sample_points", "Sample points", "input", {}),
        ("sorting_strategy", "Sorting strategy (Quadtron)", "input", {}),
        ("quantization", "Quantization levels (Quadtron)", "input", {}),
        ("repr_mode", "Edge repr (Polytron)", "select",
         {"options": [("cubic_bezier", "cubic_bezier"), ("bezier", "bezier"), ("hermite", "hermite")]}),
        ("corners_per_block", "Corners/block (Polytron: 4=quad, 8=hex)", "input", {}),
    ],
    "model": [
        ("d_model", "d_model", "input", {}),
        ("n_heads", "n_heads", "input", {}),
        ("stage_layers", "stage_layers (comma-separated)", "input", {}),
        ("n_latents", "n_latents", "input", {}),
        ("dropout", "dropout", "input", {}),
        ("ffn_mult", "ffn_mult", "input", {}),
    ],
    "attention": [
        ("use_flash_attention", "Use flash attention", "checkbox", {}),
        ("sliding_window_size", "Sliding window size (0=full)", "input", {}),
    ],
    "optim": [
        ("learning_rate", "Learning rate", "input", {}),
        ("weight_decay", "Weight decay", "input", {}),
        ("batch_size", "Batch size", "input", {}),
        ("num_epochs", "Num epochs", "input", {}),
        ("early_stopping_patience", "Early stopping patience", "input", {}),
    ],
    "rl": [
        ("rl_enabled", "Enable RL objective (Quadtron only for now)", "checkbox", {}),
        ("advantage_estimator", "Advantage estimator", "select",
         {"options": [("GRPO (group_relative)", "group_relative"), ("PPO (critic)", "critic")]}),
        ("rl_curriculum_stage", "Curriculum stage", "select",
         {"options": [("vertex", "vertex"), ("face", "face"), ("row", "row"), ("mesh", "mesh")]}),
        ("rl_rollouts_per_condition", "Rollouts per condition", "input", {}),
        ("init_checkpoint", "Init checkpoint (required for RL)", "input", {}),
    ],
    "runtime": [
        ("seed", "Seed", "input", {}),
        ("precision", "Precision", "select",
         {"options": [("bf16", "bf16"), ("fp16", "fp16"), ("fp32", "fp32")]}),
        ("log_dir", "Log dir", "input", {}),
        ("save_best", "Save best checkpoint", "checkbox", {}),
        ("save_last", "Save last checkpoint", "checkbox", {}),
    ],
}

SECTION_TITLES = {
    "pipeline": "Pipeline",
    "data": "Data",
    "model": "Model",
    "attention": "Attention backend",
    "optim": "Optimization",
    "rl": "Reinforcement learning (Part B)",
    "runtime": "Runtime",
}


class ConfigForm(ScrollableContainer):
    """One scrollable form covering every FIELD_SPECS entry, grouped into
    labeled sections matching config.py's own comment groups."""

    def compose(self) -> ComposeResult:
        default = PipelineConfig()
        for section, fields in FIELD_SPECS.items():
            yield Static(f"[b]{SECTION_TITLES[section]}[/b]", classes="section-title")
            for attr, label, kind, kw in fields:
                widget_id = f"field-{attr}"
                with Horizontal(classes="field-row"):
                    yield Label(label, classes="field-label")
                    default_val = getattr(default, attr, "")
                    if kind == "select":
                        options = kw["options"]
                        value = str(default_val)
                        matched = value if value in [v for _, v in options] else options[0][1]
                        yield Select(options, value=matched, id=widget_id, classes="field-input")
                    elif kind == "checkbox":
                        yield Checkbox(value=bool(default_val), id=widget_id, classes="field-input")
                    else:
                        if attr == "stage_layers":
                            text = ",".join(str(x) for x in default_val)
                        else:
                            text = str(default_val)
                        yield Input(value=text, id=widget_id, classes="field-input")

    def read_config(self) -> PipelineConfig:
        """Builds a PipelineConfig from current widget values, using each
        field's declared type on the dataclass to coerce Input text."""
        default = PipelineConfig()
        overrides = {}
        for fields in FIELD_SPECS.values():
            for attr, _label, kind, _kw in fields:
                widget = self.query_one(f"#field-{attr}")
                if kind == "checkbox":
                    overrides[attr] = widget.value
                elif kind == "select":
                    raw = widget.value
                    overrides[attr] = int(raw) if attr == "dim" else raw
                else:
                    raw = widget.value.strip()
                    default_val = getattr(default, attr)
                    if attr == "stage_layers":
                        overrides[attr] = tuple(int(x) for x in raw.split(",") if x.strip())
                    elif isinstance(default_val, bool):
                        overrides[attr] = raw.lower() in ("1", "true", "yes")
                    elif isinstance(default_val, int):
                        overrides[attr] = int(raw) if raw else default_val
                    elif isinstance(default_val, float):
                        overrides[attr] = float(raw) if raw else default_val
                    else:
                        overrides[attr] = raw
        return PipelineConfig(**overrides)

    def load_config(self, cfg: PipelineConfig) -> None:
        for fields in FIELD_SPECS.values():
            for attr, _label, kind, _kw in fields:
                widget = self.query_one(f"#field-{attr}")
                val = getattr(cfg, attr)
                if kind == "checkbox":
                    widget.value = bool(val)
                elif kind == "select":
                    widget.value = str(val) if attr == "dim" else val
                elif attr == "stage_layers":
                    widget.value = ",".join(str(x) for x in val)
                else:
                    widget.value = str(val)


class MeshtronTUI(App):
    CSS = """
    .section-title { margin-top: 1; color: $accent; }
    .field-row { height: 3; align: left middle; }
    .field-label { width: 42; }
    .field-input { width: 1fr; }
    #dataset-picker { height: auto; margin-bottom: 1; }
    #train-controls { height: 3; }
    #log { border: solid $accent; height: 1fr; }
    #hash-line { color: $text-muted; }
    """
    TITLE = "Meshtron"
    SUB_TITLE = "Quadtron / Polytron pipeline control"

    def __init__(self):
        super().__init__()
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._train_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Config", id="tab-config"):
                with Vertical():
                    with Horizontal(id="dataset-picker"):
                        yield Label("Quick-pick dataset: ")
                        yield Select(
                            [(f, f) for f in _discover_datasets()] or [("(none found in cwd)", "")],
                            id="dataset-select", allow_blank=True,
                        )
                    yield ConfigForm(id="config-form")
                    with Horizontal(id="config-buttons"):
                        yield Button("Save Config...", id="btn-save")
                        yield Button("Load Config...", id="btn-load")
                        yield Static("", id="hash-line")
            with TabPane("Train", id="tab-train"):
                with Vertical():
                    with Horizontal(id="train-controls"):
                        yield Button("Start Training", id="btn-start", variant="success")
                        yield Button("Stop", id="btn-stop", variant="error", disabled=True)
                    yield RichLog(id="log", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.2, self._drain_log_queue)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "dataset-select" and event.value:
            self.query_one("#field-data_path", Input).value = str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self._save_config()
        elif event.button.id == "btn-load":
            self._load_config()
        elif event.button.id == "btn-start":
            self._start_training()
        elif event.button.id == "btn-stop":
            self._stop_training()

    def _save_config(self) -> None:
        try:
            cfg = self.query_one(ConfigForm).read_config()
        except Exception as e:
            self.query_one("#hash-line", Static).update(f"[red]Config error: {e}[/red]")
            return
        path = Path(f"tui_config_{cfg.hash()}.json")
        cfg.to_json(path)
        self.query_one("#hash-line", Static).update(f"Saved {path} (hash {cfg.hash()})")

    def _load_config(self) -> None:
        candidates = sorted(Path(".").glob("tui_config_*.json"))
        if not candidates:
            self.query_one("#hash-line", Static).update("[yellow]No tui_config_*.json found[/yellow]")
            return
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        cfg = PipelineConfig.from_dict(json.loads(latest.read_text()))
        self.query_one(ConfigForm).load_config(cfg)
        self.query_one("#hash-line", Static).update(f"Loaded {latest}")

    def _start_training(self) -> None:
        if self._train_thread and self._train_thread.is_alive():
            return
        try:
            cfg = self.query_one(ConfigForm).read_config()
        except Exception as e:
            self._log_queue.put(f"[red]Config error, not starting: {e}[/red]")
            return

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        log = self.query_one("#log", RichLog)
        log.clear()
        self._log_queue.put(f"Starting {cfg.model_family} dim={cfg.dim} "
                            f"(config hash {cfg.hash()})...")

        self._stop_event.clear()

        def _run():
            from meshtron.training.trainer import TrainingCancelled
            writer = _QueueWriter(self._log_queue)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = writer, writer
            try:
                if cfg.model_family == "quadtron":
                    from meshtron.training.trainer import Trainer

                    def on_epoch(epoch, train_metrics, val_metrics):
                        if self._stop_event.is_set():
                            raise TrainingCancelled("stopped from TUI")

                    Trainer(cfg).run(on_epoch=on_epoch)
                elif cfg.model_family == "polytron":
                    from polytron_chain import run_polytron
                    run_polytron(cfg, stop_check=self._stop_event.is_set)
                else:
                    print(f"Unknown model_family={cfg.model_family!r}")
            except TrainingCancelled:
                print("Training cancelled.")
            except Exception as e:
                print(f"TRAINING FAILED: {type(e).__name__}: {e}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self._log_queue.put("__DONE__")

        self._train_thread = threading.Thread(target=_run, daemon=True)
        self._train_thread.start()

    def _stop_training(self) -> None:
        # Cooperative: sets a flag Trainer's on_epoch callback / run_polytron's
        # stop_check poll between epochs/stages and raise TrainingCancelled on.
        # Coarser than instant (can't interrupt mid-epoch/mid-stage -- see
        # run_polytron's own stop_check docstring for why), but real, unlike
        # the button just resetting the UI while the thread kept running.
        self._stop_event.set()
        self.query_one("#btn-stop", Button).disabled = True
        self._log_queue.put("[yellow]Stop requested -- will cancel at the next "
                            "epoch/stage boundary.[/yellow]")

    def _drain_log_queue(self) -> None:
        log = self.query_one("#log", RichLog)
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            if line == "__DONE__":
                self.query_one("#btn-start", Button).disabled = False
                self.query_one("#btn-stop", Button).disabled = True
                log.write("[green]Training finished.[/green]")
            else:
                log.write(line)


if __name__ == "__main__":
    MeshtronTUI().run()
