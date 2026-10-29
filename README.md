# Replication package: A Calibrated Placebo Test for Time-Varying Features in Credit Risk Models

Author: Muhammad Nawaz. Companion to SSRN 7622180 (https://ssrn.com/abstract=7622180).

## What each file does (plain language)
| File | What it does |
|---|---|
| `sim.py` | Builds one simulated loan portfolio, runs the standard check and the two placebo tests (year effects included). |
| `sim2.py` | Same, with options for the credit-cycle strength and for switching year effects off. |
| `run_sim.py` | Runs the 9 main settings, 300 repetitions each -> `sim_results.csv`. |
| `run_sim2.py` | Runs the 5 extra settings (no period control, drifting cycle) -> `sim2_results.csv`. |
| `summarise.py` | Turns the raw results into pass rates with 95% Wilson intervals -> `sim_summary.csv`. |
| `make_figures.py` | Draws Figure 1 and Figure 2 from `sim_summary.csv`. |
| `real.py` | Real-data illustration (Lending Club with FOMC tone, Bondora with ECB tone). Needs the data and tone files of the companion paper; set `PAPER1_DIR` to that folder. Usage: `python3 real.py {lc|bondora} {none|year|quarter} 99 60000 out.csv` |
| `real_results.csv` | The six real-data results used in Table 3. |

## Run order
```
pip install numpy scipy pandas scikit-learn joblib matplotlib
python3 run_sim.py 300      # slow: several hours on 2 cores
python3 run_sim2.py 300
python3 summarise.py
python3 make_figures.py
```
The raw result files from our run are included, so `summarise.py` and `make_figures.py` can be run at once.

## Notes
- Placebo test passes when the p-value is below 0.05, with 99 placebos.
- The Lending Club and Bondora data are not redistributed here; see the companion paper's repository for sources and terms.
- Seeds are fixed in the scripts, so results should reproduce.

## Quick check (about 10 minutes)
Copy this folder first, because the scripts overwrite the result files. Then:
```
python3 sim.py                # one simulated run; prints the three test results
python3 run_sim.py 5          # 5 repetitions per setting instead of 300
```
Results from 5 repetitions are noisy; only the 300-repetition run reproduces the paper.

## Disclosure of AI use
Claude (Anthropic), a generative AI assistant, was used for language correction, for searching relevant literature, and for drafting and correcting the Python code. The author ran and checked the code and the results and is responsible for the content.
