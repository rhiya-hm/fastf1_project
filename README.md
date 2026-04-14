# F1 Data Analysis Project

A personal data analysis project built around the [FastF1](https://docs.fastf1.dev/) Python library, exploring Formula 1 race data across the past six seasons. The project covers race results, lap-time analysis, qualifying telemetry, tyre strategy, and machine-learning-based race predictions — all visualised with dark-themed matplotlib charts.

This project was built to strengthen Python data analysis skills and produce portfolio-ready outputs for a data science career path. Development was assisted by [Claude](https://claude.ai) (Anthropic).

---

## Prerequisites

**Python 3.10+** is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `fastf1` | F1 telemetry, lap, and results data via the official API |
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations and interpolation |
| `matplotlib` | All charting and visualisation |
| `scikit-learn` | Random Forest model for race predictions |

A local cache folder (`Cache/`) is created automatically on first run to avoid re-fetching data.

---

## Project Structure

```
fastf1_project/
├── main.py                  # Entry point — prints 2026 race winners to date
├── lap_by_lap.py            # Lap-time table and race charts for Qatar GP 2025
├── predictions.py           # Random Forest model predicting race finishing order
├── miami_predictor.py       # Focused prediction for the 2026 Miami Grand Prix
├── telemetry_compare.py     # Qualifying telemetry comparison (track map + speed trace)
├── tyre_strategy.py         # Full 2025 season tyre strategy analysis (3 charts)
├── plot_laptimes.py         # Placeholder — lap time plotting
├── explore_session.py       # Placeholder — session exploration
├── telemetry_compare_files/ # Output folder for telemetry charts
├── tyre_strategy_files/     # Output folder for tyre strategy charts
├── Cache/                   # FastF1 local cache (auto-generated)
└── requirements.txt
```

---

## Scripts

### `main.py`
Iterates all completed conventional races in the 2026 season and prints a table of race winners. Stops automatically at any race that hasn't happened yet using UTC timestamps.

### `lap_by_lap.py`
Loads the 2025 Qatar Grand Prix race session and produces:
- A printed table of best lap and average lap time per driver (outlier laps filtered at 115% of personal best)
- A dual-chart figure: lap times over the race for all drivers, and a fastest-lap bar chart per driver

### `predictions.py`
Trains a **Random Forest Regressor** on 2022–2025 historical race results (grid position, driver, team, recent form) and predicts the finishing order for the 2026 Miami Grand Prix. Features include rolling 3-race form averages and circuit-specific flags. Outputs model accuracy (MAE) and feature importances alongside the predicted grid.

### `miami_predictor.py`
A standalone Miami-focused predictor producing the same lap-time table and charts as `lap_by_lap.py` but scoped to Qatar 2025 as a base dataset.

### `telemetry_compare.py`
Compares the qualifying telemetry of the top two drivers at a given race (default: 2026 Chinese GP). Produces a dark-themed three-panel figure:
- **Individual speed maps** — the circuit coloured by each driver's speed using a plasma gradient
- **Delta track map** — green where Driver 1 is faster, red where Driver 2 is faster
- **Speed trace overlay** — both drivers on the same axes with shaded advantage regions

Output is saved to `telemetry_compare_files/`. Change `YEAR`, `EVENT`, or driver indices at the top of the file to compare any session.

### `tyre_strategy.py`
Loads all conventional races from the 2025 season and produces three charts saved to `tyre_strategy_files/`:

1. **`01_strategy_timeline_{Race}_{Year}.png`** — per-driver Gantt-style stint map for a chosen showcase race, coloured by compound and sorted by finishing position
2. **`02_compound_degradation.png`** — aggregate lap-time delta vs tyre age for soft/medium/hard across the full season, with ±1 std dev bands
3. **`03_pit_timing.png`** — first pit stop timing (as % of race distance) vs final position across all races, plus average finish position by number of stops

Change `SHOWCASE_RACE` at the top of the file to regenerate chart 1 for any race without affecting charts 2 and 3.

---

## Running the Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Race winners so far this season
python main.py

# Lap time analysis (Qatar 2025)
python lap_by_lap.py

# Race outcome predictions (Miami 2026)
python predictions.py

# Qualifying telemetry comparison
python telemetry_compare.py

# Full season tyre strategy analysis
python tyre_strategy.py
```

> **Note:** First-run data fetching can take several minutes depending on the session. All data is cached locally in `Cache/` for fast repeat runs.

---

## What I Learned

This project was built while relearning Python for data science. Key concepts and skills developed during the build:

**Data handling**
- Loading and filtering large DataFrames with pandas (lap filtering, outlier removal, groupby aggregations)
- Merging datasets across multiple sources (results, laps, telemetry)
- Working with time-series data including `timedelta` lap times and UTC-aware `datetime` objects

**Visualisation**
- Building multi-panel dark-themed figures with `matplotlib.gridspec`
- Colouring line segments by a continuous variable using `LineCollection` — used for the track map speed overlays
- Applying diverging colour scales (`TwoSlopeNorm`) to show directional deltas (e.g. which driver is faster at each point on track)
- Interpolating two datasets onto a common distance axis with `numpy.interp` for fair comparisons

**Machine learning**
- Training a `RandomForestRegressor` from scikit-learn on structured historical data
- Feature engineering: rolling averages, label encoding, circuit-specific binary flags
- Evaluating model performance with MAE and inspecting feature importances

**API and caching patterns**
- Using the FastF1 library to fetch session data, lap data, and per-driver telemetry
- Enabling local disk caching to avoid redundant API calls across runs
- Filtering race schedules by event format (`conventional` vs `sprint_shootout`)

**Development approach**
- Structuring a multi-script analysis project with clear separation of concerns
- Using [Claude](https://claude.ai) (Anthropic) as an AI development assistant to design scripts, debug logic, and produce production-quality visualisations — learning how to collaborate effectively with AI tooling as part of a real workflow

---

## Data Source

All data is sourced from the [FastF1 Python library](https://docs.fastf1.dev/), which provides access to the official Formula 1 timing and telemetry API. Data is available from the 2018 season onwards.
