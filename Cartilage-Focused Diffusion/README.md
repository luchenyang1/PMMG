# Cartilage-Focused Diffusion

Standalone module for **cartilage-focused MRI synthesis**. Trained and inferred separately from PMMG; generated volumes are placed into `../dataroot/` for PMMG training.

**Pipeline:** train here → inference → copy outputs to PMMG dataroot → `python ../run.py`

| `data_type` | PMMG modality |
|-------------|---------------|
| `pfoa` | base PDW setup |
| `pfoa_t1w` | `sag T1w_tse` |
| `pfoa_pdw_atse` | `sag PDW_atse` |

---

## Data format

Training `root_dir`:
```
root_dir/
├── images/   # *.nii.gz
└── labels/   # matching masks
```

PMMG expects outputs at:
```
../dataroot/center1/<case_id>/<modality>/image.nii.gz
```

---

## Train

From `train/` (Hydra configs in `train/config/`):

```bash
cd train
python train.py dataset=pfoa dataset.root_dir=/path/to/data model.results_folder=/path/to/checkpoints
```

Replace `???` in YAML files or override on the command line.

---

## Inference

From `inference/`:

```bash
cd inference
python inference.py \
  data_type=pfoa \
  model_path=/path/to/checkpoint.pt \
  dataset_root_dir=/path/to/test_data \
  target_img_path=/path/to/output/images \
  target_label_path=/path/to/output/labels
```

Requires `inference/hist_clusters/<data_type>_clusters.json`.
