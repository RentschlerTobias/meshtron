# 3D Smoke Result — meshtron local training environment

Status: **smoke passed end-to-end** for both Quadtron and Polytron on the
`data/hex3d_algohex/smoke2000/` sample.

---

## 1. Environment

| Item | Value |
|---|---|
| Host python | 3.14.7 (system `/usr/bin/python3`) |
| uv-managed python | 3.12.13 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| torch_geometric | 2.8.0.post1 |
| CUDA available | True |
| CUDA version (PyTorch) | 12.8 |
| cuDNN | 91900 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU compute capability | 8.9 |
| Total GPU memory | 8188 MiB |
| bf16 supported | Yes (compute 8.9) |

Verification command (run inside `uv run`):

```python
import torch, torch_geometric
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch_geometric.__version__)
```

---

## 2. Converter output (`domain_extractor_3d.py`)

The converter expects `--src` to point to a **parent** directory that contains
one sub-folder per sample (`*/sample.npz`). For the smoke sample the working
invocation is therefore:

```bash
uv run python domain_extractor_3d.py \
  --src data/hex3d_algohex/ \
  --out-quadtron /tmp/opencode/meshtron-smoke/quadtron_data_3d.pt \
  --out-polytron /tmp/opencode/meshtron-smoke/polytron_data_3d.pt \
  --max-tri-points 768
```

It found exactly one `sample.npz` under `data/hex3d_algohex/` and extracted
1/1 samples.

Source `sample.npz` summary (15 keys):

| Key | Shape | Dtype |
|---|---|---|
| `blocks` | (16, 8) | int64 |
| `dir_class` | (214,) | int64 |
| `dir_class_count` | (11,) | int64 |
| `edge_ctrl` | (214, 2, 3) | float64 |
| `edge_polyline` | (1270, 3) | float64 |
| `edge_polyline_offset` | (215,) | int64 |
| `edges` | (214, 2) | int64 |
| `quad_faces` | (50, 4) | int64 |
| `surface_points` | (11926, 3) | float64 |
| `surface_tri_label` | (18548,) | int64 |
| `surface_tris` | (18548, 3) | int64 |
| `vertices` | (50, 3) | float64 |

Measured PolytronTokenizer bounds printed by the converter:

```text
r_bounds=(0.5524843335151672, 1.8283976316452026)
z_bounds=(-0.04999999701976776, 2.549999952316284)
```

### 2.1 Quadtron format

Type: `list[torch_geometric.data.Data]` with length 1.

| Attribute | Shape | Dtype |
|---|---|---|
| `x` | (50, 3) | float32 |
| `faces` | (4, 50) | int64 |
| `tri_coordinates` | (11926, 3) | float32 |
| `dir_class` | (50,) | int64 |

Tokenization check (dim=3, strategy 0):

```text
vocab_size=260, tokens_per_face=12, sequence_length=616
```

### 2.2 Polytron format

Type: `list[dict]` with length 1.

| Key | Type / Shape | Dtype |
|---|---|---|
| `vertices_polar` | (50, 3) | float32 |
| `vertices_cartesian` | (50, 3) | float32 |
| `faces` | (8, 16) | int64 |
| `edge_index` | (2, 214) | int64 |
| `edge_ctrl` | (214, 2, 3) | float32 |
| `edge_to_streamline` | dict, 214 entries | — |
| `center` | (3,) | float32 |
| `quad_faces` | (4, 50) | int64 |
| `tri_coordinates` | (768, 3) | float32 (subsampled from 11926) |
| `surface_points` | (11926, 3) | float32 (full surface) |

---

## 3. Training smoke

The single smoke sample was repeated 6× in a separate temporary `.pt` file so
that Polytron's hard-coded 10 % validation split (`n_val = max(1, N//10)`)
produces a non-empty training set (5 train / 1 val). Quadtron also benefits
from the same split. The repetition does not add new geometry; it is only a
workaround for the degenerate 1-sample case.

### 3.1 Quadtron

Command (see `scripts/3d_pipeline_smoke.sh`):

```bash
uv run python train.py \
  --model-family quadtron --dim 3 \
  --data-path /tmp/opencode/meshtron-smoke/quadtron_data_3d_6x.pt \
  --d-model 64 --n-heads 2 --stage-layers 2 2 2 --n-latents 16 \
  --batch-size 1 --num-epochs 1 --precision bf16 --quantization 256 \
  --log-dir /tmp/opencode/meshtron-smoke/runs --save-last
```

Results:

| Metric | Value |
|---|---|
| Train samples | 5 |
| Val samples | 1 |
| Train steps | 5 |
| Train bpt start → end | 8.245 → 8.097 |
| Val bpt | 7.761 |
| Steps/sec (train) | ~4.1 it/s |
| Epochs/sec | ~3.2 epochs/s |
| Wall time (1 epoch) | 0.42 s |
| Peak GPU memory | **78.3 MiB** |

### 3.2 Polytron

Command (see `scripts/3d_pipeline_smoke.sh`):

```bash
uv run python train.py \
  --model-family polytron --dim 3 \
  --data-path /tmp/opencode/meshtron-smoke/polytron_data_3d_6x.pt \
  --d-model 64 --batch-size 1 --num-epochs 1 --precision bf16 \
  --corners-per-block 8 --repr-mode cubic_bezier \
  --log-dir /tmp/opencode/meshtron-smoke/runs
```

Results:

| Head | Train loss | Val loss |
|---|---|---|
| S1 (vertices) | 7.2761 | 6.5385 |
| S2 (pointer faces) | 3.9267 | 3.8487 |
| S3 (HO geometry) | 0.1987 | 0.0521 |

End-to-end chain inference and gallery plot also completed.

| Metric | Value |
|---|---|
| Train samples | 5 |
| Val samples | 1 |
| Train steps per head | 5 |
| Wall time (1 epoch, 3 heads + chain) | 1.48 s |
| Peak GPU memory | **68.3 MiB** |
| Gallery output | `figures/e2e/e2e_gallery.png` |

---

## 4. Blockers found and fixes applied

All fixes are minimal, local, and only touch files that were blocking the 3D
smoke. They are **not** committed.

### 4.1 openmesh build failure blocked Quadtron import

**Symptom:** `uv sync` works, but `from tokenizer_v2 import Tokenizer2D` and
`from trainer import Trainer` raised:

```text
ModuleNotFoundError: No module named 'openmesh'
```

**Root cause:** `tokenizer_v2.py` imported `openmesh as om` at the top level
(dead import) and `from half_edge import order_quads_yx`; `half_edge.py`
imports `openmesh` unconditionally. The `openmesh` extra is optional and its
sdist (`openmesh==1.2.1`) fails to build with the system CMake 4.4.2 because
its `CMakeLists.txt` declares a minimum version below 3.5:

```text
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

**Fix:**

- `tokenizer_v2.py`: removed the unused `import openmesh as om`.
- `half_edge.py`: guarded `import openmesh as om` with `try/except` and added
  a clear `RuntimeError` inside `order_quads_yx()` if openmesh is unavailable.

The dim=3 Quadtron path never calls `order_quads_yx`, so the smoke now runs
without building openmesh. dim=2 paths that need half-edge ordering will still
raise a helpful error if openmesh is missing.

### 4.2 Polytron gallery crashed with a single facecount

**Symptom:** `run_polytron()` completed S1/S2/S3 training, then crashed during
gallery generation:

```text
File "polytron_chain.py", line 337, in run_polytron
    draw_gen(axes[1, c], ...)
IndexError: index 1 is out of bounds for axis 0 with size 1
```

**Root cause:** `plt.subplots(2, 1)` squeezes the returned `axes` to shape
`(2,)`, and `np.atleast_2d(axes)` turned it into `(1, 2)`.

**Fix:** `polytron_chain.py`: added `squeeze=False` to `plt.subplots` so the
axes array is always 2D.

### 4.3 True 1-sample Polytron smoke has empty train split

**Symptom:** When running Polytron on the original 1-sample `.pt`,
`run_polytron()` splits into `0 train / 1 val`; `train_pointer.py` then divides
by zero:

```text
File "train_pointer.py", line 96, in run_epoch
    return tot_loss / tot_tok, tf_correct / tot_tok
ZeroDivisionError: float division by zero
```

**Workaround:** The smoke script duplicates the single converted sample 6× in a
temporary `.pt` file. This is **not** a code change; it gives Polytron a
non-empty train split (5/1) so the smoke actually executes training steps.

---

## 5. Realistic full-dataset config for 8 GB

Measured on the 6×/20× repeated smoke sets (same geometry as the real sample,
so the memory footprint per sample is representative). bf16 is enabled on the
RTX 4060.

### 5.1 Quadtron

| d_model | batch_size | Flash + window 64 | Peak VRAM | Fits 8 GB? |
|---|---|---|---|---|
| 64 | 1 | no | 78 MiB | yes |
| 512 | 1 | yes | 1787 MiB | yes |
| 512 | 8 | yes | 4395 MiB | yes (recommended) |
| 512 | 16 | yes | 6721 MiB | yes, tight |

Recommended starting point for ~500 samples on this GPU:

```bash
--model-family quadtron --dim 3 --d-model 512 --n-heads 8 \
--stage-layers 8 8 8 --n-latents 64 --batch-size 8 \
--precision bf16 --use-flash-attention --sliding-window-size 64
```

Expected peak VRAM: **~4.4 GB**.

### 5.2 Polytron

| d_model | batch_size | Peak VRAM | Fits 8 GB? |
|---|---|---|---|
| 64 | 1 | 68 MiB | yes |
| 512 | 8 | 2098 MiB | yes (recommended) |
| 1024 | 8 | 5470 MiB | yes |

Recommended starting point:

```bash
--model-family polytron --dim 3 --d-model 512 --batch-size 8 \
--precision bf16 --corners-per-block 8 --repr-mode cubic_bezier
```

Expected peak VRAM: **~2.1 GB**.

### 5.3 Wall-clock estimate per epoch for ~500 samples

Measured step times were used to extrapolate to 500 samples.

**Quadtron** (`d_model=512, batch_size=8, flash+window 64`):

- Observed: ~0.37 s per training step (batch of 8).
- 500 samples / 8 ≈ 63 steps.
- Estimated training time per epoch: **~25–30 s**.

**Polytron** (`d_model=512, batch_size=8`):

- Observed: ~0.5 s per training step for S1 (the slowest head).
- 500 samples / 8 ≈ 63 steps per head.
- Estimated training time per epoch for all three heads: **~60–90 s**.
- Note: this is training time only; chain inference on held-out samples is
  autoregressive and scales linearly with `eval_n`.

---

## 6. Deliverables

- `scripts/3d_pipeline_smoke.sh` — copy-pasteable, runs from repo root.
- This report: `reports/3d_smoke_result.md`.

No full-dataset `.pt` files were generated; all artifacts live under
`/tmp/opencode/meshtron-smoke/`.
