# PMMG

Multi-modal, multi-view **PFOA cartilage defect grading** from five MRI sequences, clinical features, and radiology text.

This repo has two **independent** parts:

| Part | Location | Role |
|------|----------|------|
| PMMG classifier | repo root | Train / test the grading model (`run.py`) |
| Cartilage-Focused Diffusion | `Cartilage-Focused Diffusion/` | Optional MRI synthesis; see [its README](Cartilage-Focused%20Diffusion/README.md) |

**Pipeline:** (optional) train & run diffusion → put generated volumes into `./dataroot/` → `python run.py`

PMMG does not call the diffusion code at runtime; both stages share the same data layout.

---

## Layout

```
PMMG/
├── run.py              # train / test entry point
├── run/                # Args, PathDict, train loop
├── model/              # Multi_view_Knee
├── data/               # dataloader
├── Cartilage-Focused Diffusion/
├── dataroot/           # data (create before running)
├── pretrain/           # optional ResNet weights
└── finetuned_m3d_clip/ # optional M3D-CLIP weights
```

Run all commands from the repo root.

---

## Data

Configure paths in `run/PathDict.py`. Each case needs five modalities under `dataroot/center*/<case_id>/`:

- `sag PDW_spair`, `cor PDW_spair`, `axi T2w_tse`, `sag T1w_tse`, `sag PDW_atse`
- each folder contains `image.nii.gz`

Label CSVs (`train_center1.csv`, etc.) provide case ID (col 0), P/F grades (cols 1–2), clinical features (cols 3, 7), and report text (col 9).

Task: patellar (P) and femoral (F) grading, 4 levels each (8 targets total).

---

## Usage

**Train**
```bash
python run.py --gpu 0 --epochs 100 --batch_size 4 --lr 5e-5
```

**Test**
```bash
python run.py --test --weight_path /path/to/best.pth --gpu 0
```

Checkpoints and logs go to `./ExpFolder/`. More options in `run/Args.py`.

> `run.py` may prepend default CLI args via `sys.argv`; edit or remove that block if needed.

Default test split is `center3` (change in `run.py`).
