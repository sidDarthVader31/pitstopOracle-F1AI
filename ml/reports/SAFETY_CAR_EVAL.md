# Safety Car / VSC probability diagnostic

Source: OpenF1 historical race-control messages (2023 onward). Labels count deployed SC/VSC messages; ending messages do not count as deployment.
The model is a race-level logistic regression using only prior circuit SC/VSC rates, sourced street-circuit type, round, and regulation era. It is an experimental diagnostic and is not promoted or integrated into the race simulator.

- Races labeled: 89
- SC events: 40; VSC events: 31

## Safety Car

- Walk-forward races: 59
- Positive rate: 39.0%
- Model log loss / Brier: 0.861 / 0.308
- Expanding prevalence baseline log loss / Brier: 0.699 / 0.253
- PR-AUC / ROC-AUC: 0.349 / 0.397

## Virtual Safety Car

- Walk-forward races: 59
- Positive rate: 32.2%
- Model log loss / Brier: 0.736 / 0.262
- Expanding prevalence baseline log loss / Brier: 0.642 / 0.224
- PR-AUC / ROC-AUC: 0.340 / 0.509

No model artifact is written. Keep SC/VSC probabilities out of the production simulator unless the walk-forward scores and calibration justify their use.
