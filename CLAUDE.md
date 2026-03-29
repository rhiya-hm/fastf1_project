# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Architecture

This is a multi-script F1 data analysis project using the [FastF1](https://docs.fastf1.dev/) library with pandas and matplotlib.

- **`main.py`** — entry point; currently iterates 2026 conventional races and prints race winners
- **`plot_laptimes.py`**, **`telemetry_compare.py`**, **`explore_session.py`** — placeholder scripts for plotting and exploration
- **`Cache/`** — FastF1 local cache directory (enabled via `fastf1.Cache.enable_cache('cache')`). Session data is stored as `.ff1pkl` files organized by `year/event/session/`. This cache avoids re-fetching data from the F1 API on repeat runs.

### Key FastF1 patterns

```python
# Load a specific session (year, event name or round number, session type)
session = fastf1.get_session(2025, 'Monaco Grand Prix', 'Q')  # Q=Qualifying, R=Race, FP1/FP2/FP3
session.load()  # fetches all data (laps, results, telemetry, weather)
# Selectively load only what's needed (faster):
session.load(laps=True, telemetry=False, weather=False, messages=False)

# Access results, laps, telemetry
session.results          # DataFrame: finishing positions, driver/team info
session.laps             # DataFrame: all laps with timing data
session.laps.pick_driver('HAM').get_telemetry()  # per-driver telemetry

# Get event schedule for a season
schedule = fastf1.get_event_schedule(2025)
# EventFormat values: 'conventional', 'sprint_shootout', etc.
# Filter to only conventional (non-sprint) weekends:
schedule = schedule[schedule['EventFormat'] == 'conventional']
```

Session types: `'FP1'`, `'FP2'`, `'FP3'`, `'Q'` (Qualifying), `'SQ'` (Sprint Qualifying), `'S'` (Sprint), `'R'` (Race).

### Plotting

Use `fastf1.plotting` helpers alongside matplotlib. FastF1 provides team colors and driver color maps:

```python
import fastf1.plotting
fastf1.plotting.setup_mpl()  # applies F1-themed matplotlib style
```
