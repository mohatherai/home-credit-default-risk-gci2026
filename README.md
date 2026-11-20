<img width="1181" height="687" alt="image" src="https://github.com/user-attachments/assets/cac7bad7-f8cc-4dda-86c0-b773e9202f2a" />
# Home Credit Default Risk – GCI World 2026 Competition

Predict loan default (binary target) from the provided `train.csv` / `test.csv`. Metric: ROC-AUC.

## Approach
- Data: only the provided files (no external data, no hand-labeling).
- Cleaning: `DAYS_EMPLOYED = 365243` → NaN with an anomaly flag; `OWN_CAR_AGE >= 60` and `ORGANIZATION_TYPE = XNA` treated as missing.
- Features (87): lender-style ratios, time ratios relative to age, `EXT_SOURCE` aggregates and interactions, value-frequency counts.
- Models: LightGBM (3 seeds), CatBoost, XGBoost, each with 5-fold stratified CV and early stopping.
- Ensemble: rank average; weights chosen on out-of-fold AUC only (never the leaderboard).
- Reproducibility: all seeds fixed; Restart & Run all reproduces `output/submission.csv`.

## Results
| Model | OOF AUC |
|---|---|
| LightGBM (3 seeds) | 0.76430 |
| CatBoost | 0.76355 |
| XGBoost | 0.76293 |
| Blend (0.4 / 0.4 / 0.2) | 0.76567 |

Public leaderboard score: 0.7681.

## Run
1. Put `train.csv`, `test.csv`, `sample_submission.csv` in an `input/` folder next to the notebook (the data is not included in this repo).
2. `pip install -r requirements.txt`
3. Open `tutorial_final_v2.ipynb` and run all cells. On Colab it mounts Google Drive and expects the folder `MyDrive/GCI/Competition`.

## Credits
Several feature ideas (income left, overpay ratio, EXT interactions, value-frequency counts) were adapted from [diwanparker/gci-world-competition](https://github.com/diwanparker/gci-world-competition) and common public Home Credit Kaggle notebooks.
The experiments and the CV / blending code were done with the help of Claude (AI assistant) and reviewed by me.
