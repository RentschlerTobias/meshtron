from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
import json


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for Quadtron + Polytron, 2D + 3D.

    Replaces the old `TrainingConfig` (kept the same hash/JSON contract so
    existing `runs/<hash>/` tooling keeps working). `model_family` and `dim`
    select which tokenizer/model/dataset combination `train.py` builds; fields
    below are grouped by which part of the pipeline reads them, not by family
    -- unused fields for a given family are simply ignored, same as any other
    dataclass default.
    """

    # Pipeline-Auswahl
    model_family: str = "quadtron"   # "quadtron" | "polytron"
    dim: int = 2                     # 2 | 3

    # Daten
    data_path: str = "./centered_blades_cleaned.pt"
    train_val_ratio: float = 0.8
    sorting_strategy: int = 1        # Quadtron: _order_quads strategy (0-3)
    quantization: int = 256
    n_sample_points: int = 1000

    # Polytron-spezifisch (ignoriert wenn model_family="quadtron")
    repr_mode: str = "cubic_bezier"  # "hermite" | "bezier" | "cubic_bezier"
    corners_per_block: int = 4       # 4 = 2D-Quad, 8 = 3D-Hex-Block

    # Modell
    d_model: int = 512
    n_heads: int = 8
    stage_layers: tuple = (8, 8, 8)
    n_latents: int = 64
    dropout: float = 0.1
    ffn_mult: int = 4

    # Attention-Backend (unabhaengige Flags, 0 = aus wie max_val_batches)
    use_flash_attention: bool = False
    sliding_window_size: int = 0     # 0 -> volle Attention, >0 -> Fenstergroesse (Tokens)

    # Optimierung
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    batch_size: int = 16
    accumulation_steps: int = 1
    num_epochs: int = 15
    early_stopping_patience: int = 25
    max_val_batches: int = 0  # 0 -> ganzer Val-Loader

    # Validierung
    val_every_n_epochs: int = 1

    # Reinforcement Learning (Teil B, nur wenn rl_enabled -- siehe
    # docs/decisions/2026-09-13-meshtron-refactor-plan.md)
    rl_enabled: bool = False
    advantage_estimator: str = "group_relative"  # "group_relative" (GRPO) | "critic" (PPO)
    rl_rollouts_per_condition: int = 8
    rl_curriculum_stage: str = "vertex"  # "vertex" | "face" | "row" | "mesh"
    reward_weights: dict = field(default_factory=dict)
    rl_temperature: float = 1.0
    rl_max_length: int = 0  # 0 -> use the batch's own padded length as the rollout cap

    # Checkpoint, um EIN vortrainiertes Modell weiterzutrainieren (RL faengt nie bei
    # Zufallsgewichten an -- ein zufaelliges Policy hat ~0 Reward auf Struktur-
    # validitaets-Rewards, siehe A+C-Gate in der Decision-Log). "" = kein Laden
    # (normales Teacher-Forcing-Training von Grund auf).
    init_checkpoint: str = ""

    # Laufzeit
    seed: int = 0
    precision: str = "bf16"  # "fp32" | "bf16" | "fp16"
    cudnn_deterministic: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    # Persistenz / Logging
    log_dir: str = "runs"
    save_best: bool = False
    save_last: bool = False

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return sha1(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        if "stage_layers" in data and isinstance(data["stage_layers"], list):
            data = {**data, "stage_layers": tuple(data["stage_layers"])}
        return cls(**data)


@dataclass(frozen=True)
class DomainTrainingConfig:
    """Konfiguration für Domain-Partition Training."""

    # Daten
    data_path: str = "./domain_data_10k.pt"
    train_val_ratio: float = 0.8
    sorting_strategy: int = 0       # 0=no compression, 1=row-compressed, 2=vertex-first
    embedding_mode: int = 0          # 0=split vocab, 1=shared, 2=separate
    quantization_r: int = 512
    quantization_a: int = 256
    n_sample_points: int = 768       # < min Punktzahl (372..819) im 10k-Datensatz
    point_cloud_labels: bool = True  # Label {0=Ecke,1=Rand,2=Feld} an Punktwolke
    #                                  anhaengen + Ecken garantiert behalten
    #                                  (Ecken = exakte Vertices, siehe Stufe-1-Befund)

    # Modell
    d_model: int = 512
    n_heads: int = 8
    stage_layers: tuple = (2, 4, 6, 8, 10)
    n_latents: int = 512
    dropout: float = 0.1
    ffn_mult: int = 4

    # Optimierung
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    batch_size: int = 8
    accumulation_steps: int = 1
    num_epochs: int = 50
    early_stopping_patience: int = 25
    max_val_batches: int = 0

    # Validierung
    val_every_n_epochs: int = 1

    # Laufzeit
    seed: int = 0
    precision: str = "bf16"
    cudnn_deterministic: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    # Persistenz / Logging
    log_dir: str = "runs_domain"
    save_best: bool = True
    save_last: bool = False

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return sha1(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_dict(cls, data: dict) -> "DomainTrainingConfig":
        if "stage_layers" in data and isinstance(data["stage_layers"], list):
            data = {**data, "stage_layers": tuple(data["stage_layers"])}
        return cls(**data)
