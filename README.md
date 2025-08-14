# Automatic Classification of Water Drop Impact Characteristics Using Audio

This repository accompanies the manuscript:

> **Automatic Classification of Water Drop Impact Characteristics Using Audio Information**  
> Merav Arogeti, Etan Fisher, and Dima Bykhovsky

The work investigates whether **impact velocity** and **drop volume** can be inferred from **audio** recordings of water-drop impacts using modern signal-processing and machine-learning methods.

The repository includes:
- Dataset of water-drop impact audio recordings
- Code to reproduce all the results reported in the paper
- Supplementary results and plots beyond these in the paper

---
### Dataset
The experiments operate on a pre-processed dataset serialized as `data_drops.pkl` . The file is expected to contain the following keys:

- `segments` — `float` array of shape `(N, T)` with time-domain audio segments (e.g., sampled at 44.1 kHz; each segment is aligned to an impact event).
- `s_label_data` — sequence of raw **impact velocities** for each segment (length `N`).
- `v_label_data` — sequence of raw **drop volumes** for each segment (length `N`).
- `unique_speeds` — sorted unique velocities (used to map raw labels to class indices).
- `unique_volumes` — sorted unique volumes (used to map raw labels to class indices).

### Notebook and Script Overview
- `catch22.ipynb` — catch22 features
- `minirocket.ipynb` — MiniRocket features/classifier
- `mfcc.ipynb` — MFCC spectrogram features with feature selection
- `mfcc_average.ipynb` — temporally averaged MFCCs
- `periodogram.ipynb` — PSD/periodogram features
- `rise.ipynb` — Random Interval Spectral Ensemble (RISE)
- `spectrogram.ipynb` — STFT features with feature selection
- `scattering_tr.ipynb` — Scattering Transform features (Kymatio)
- `ts_fresh_comp.ipynb` — compact feature set `tsfresh`
- `ts_fresh_eff.ipynb` — full feature set from `tsfresh`

##### Scripts
- `drop_lib2.py` — library of utility functions for loading and processing the dataset
- `drop_lib3.py` — library of utility functions for loading and processing the dataset (different FE for volume and velocity)

---
