# A Calibrated Placebo Test for Time-Varying Features in Credit Risk Models

Replication code for the working paper by **Muhammad Nawaz** (October 2026).

- Paper (SSRN): link to be added after posting
- Companion paper: *Explainable Machine Learning for Basel and IFRS 9 Credit Risk: A Placebo-Controlled Test of Narrative Features*, https://ssrn.com/abstract=7622180
- Companion code: https://github.com/drnawazuum/Placebo-test-narrative-features

## What the paper is about
Banks add time-varying inputs (macro indicators, text sentiment) to credit models. The usual check is whether the model's AUC improves. In simulation, a useless but slowly changing series passes this check far more often than the intended 5%, especially when the model has no period control. A placebo test that uses fake series with the same persistence as the real one keeps the false-pass rate at 2% to 5%, at a cost in power. Two public loan datasets (Lending Club and Bondora) illustrate the point.

## Quick start
Open `Paper_A_simulation.ipynb` in Jupyter and run the cells from top to bottom.

- Default is a quick test (`REPS = 10`, a few minutes). Results are noisy.
- Set `REPS = 300` to reproduce the paper (roughly 1.5 to 2 hours on 2 CPU cores).
- Packages: `numpy scipy pandas matplotlib joblib` (`pip install numpy scipy pandas matplotlib joblib`).

## Files
| File | What it does |
|---|---|
| `Paper_A_simulation.ipynb` | All simulation code, tables and figures in one notebook |
| `sim.py`, `sim2.py` | Simulation and the three tests (standard check, shuffled placebo, persistence-matched placebo) |
| `run_sim.py`, `run_sim2.py` | Run the 14 settings of the paper |
| `summarise.py` | Pass rates with 95% Wilson intervals (`sim_summary.csv`) |
| `make_figures.py` | Figures 1 and 2 |
| `real.py`, `real_results.csv` | Real-data runs (Table 3); need the data of the companion paper, set `PAPER1_DIR` |
| `sim_results.csv`, `sim2_results.csv` | Raw results of the 300-repetition runs |

## Data
Lending Club and Bondora data are not redistributed here. See the companion repository for sources and terms of use.

## Notes
- A placebo test passes when its p-value is below 0.05 (99 placebo series).
- Seeds are fixed in the scripts.
- The simulation is deliberately simple (logistic scorecard, random train-test split). See the limitations section of the paper.

## Disclosure of AI use
Claude (Anthropic), a generative AI assistant, was used for language correction, for searching relevant literature, and for drafting and correcting the Python code. The author ran and checked the code and the results and takes full responsibility for the content.

## Citation
Nawaz, M. (2026). A calibrated placebo test for time-varying features in credit risk models. Working paper.

## Licence
MIT (add a `LICENSE` file by choosing the MIT template when you create the repository).
