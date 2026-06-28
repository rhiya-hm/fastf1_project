from datetime import datetime, timezone
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from fastf1.exceptions import RateLimitExceededError


# ── Cache setup ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("/Users/rhiyamehta/PycharmProjects/fastf1_project/Cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


# ── 1. COLLECT HISTORICAL RACE DATA (2022–2025) ──────────────────────────────
print("📦 Loading historical race data (2022–2025)... this may take a while")

all_results = []
for year in range(2022, 2026):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']  # exclude pre-season testing

    for _, event in schedule.iterrows():
        round_number = event['RoundNumber']
        event_format = event['EventFormat']

        try:
            # ── Qualifying ──
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load(telemetry=False, weather=False, messages=False)
            quali_res = quali.results[['FullName', 'Position']].rename(
                columns={'Position': 'QualiPosition'}
            )

            # ── Race ──
            race = fastf1.get_session(year, round_number, 'R')
            race.load(telemetry=False, weather=False, messages=False)
            race_res = race.results[['FullName', 'TeamName', 'GridPosition', 'Position']].copy()

            # ── Sprint (only on sprint weekends) ──
            if 'sprint' in event_format.lower():
                sprint = fastf1.get_session(year, round_number, 'S')
                sprint.load(telemetry=False, weather=False, messages=False)
                sprint_res = sprint.results[['FullName', 'Position']].rename(
                    columns={'Position': 'SprintPosition'}
                )
            else:
                sprint_res = None

            # Merge them all together
            merged = race_res.merge(quali_res, on='FullName', how='left')
            if sprint_res is not None:
                merged = merged.merge(sprint_res, on='FullName', how='left')
            else:
                merged['SprintPosition'] = np.nan

            merged['Year'] = year
            merged['RoundNumber'] = round_number
            merged['EventName'] = race.event['EventName']
            merged['EventFormat'] = event_format
            all_results.append(merged)

        except RateLimitExceededError:
            print("🛑 Rate limit hit — stopping early. Re-run in an hour to continue.")
            raise
        except Exception as e:
            print(f"  ⚠️  Skipping {year} R{round_number}: {e}")
            continue

if not all_results:
    raise RuntimeError(
        "No historical results loaded — likely a rate limit or cache issue. "
        "Check the warnings above."
    )

df = pd.concat(all_results, ignore_index=True)
print(f"  ✅ Loaded {len(df)} historical results from {df['EventName'].nunique()} events")

# ── 2. COLLECT 2026 RACE DATA (completed races only) ─────────────────────────
print("\n📦 Loading 2026 race data...")

schedule_2026 = fastf1.get_event_schedule(2026)
schedule_2026 = schedule_2026[schedule_2026['EventFormat'] != 'testing']  # exclude pre-season testing

results_2026 = []
for _, event in schedule_2026.iterrows():
    round_number = event['RoundNumber']
    event_format = event['EventFormat']

    # Stop once we hit a race that hasn't finished yet
    race_session = fastf1.get_session(2026, round_number, 'R')
    if datetime.now(timezone.utc) < race_session.event['Session5DateUtc'].replace(tzinfo=timezone.utc):
        break

    try:
        # ── Qualifying ──
        quali = fastf1.get_session(2026, round_number, 'Q')
        quali.load(telemetry=False, weather=False, messages=False)
        quali_res = quali.results[['FullName', 'Position']].rename(
            columns={'Position': 'QualiPosition'}
        )

        # ── Race ──
        race_session.load(telemetry=False, weather=False, messages=False)
        race_res = race_session.results[['FullName', 'TeamName', 'GridPosition', 'Position']].copy()

        # ── Sprint (only on sprint weekends) ──
        if 'sprint' in event_format.lower():
            sprint = fastf1.get_session(2026, round_number, 'S')
            sprint.load(telemetry=False, weather=False, messages=False)
            sprint_res = sprint.results[['FullName', 'Position']].rename(
                columns={'Position': 'SprintPosition'}
            )
        else:
            sprint_res = None

        # Merge
        merged = race_res.merge(quali_res, on='FullName', how='left')
        if sprint_res is not None:
            merged = merged.merge(sprint_res, on='FullName', how='left')
        else:
            merged['SprintPosition'] = np.nan

        merged['Year'] = 2026
        merged['RoundNumber'] = round_number
        merged['EventName'] = race_session.event['EventName']
        merged['EventFormat'] = event_format
        results_2026.append(merged)
        print(f"  ✅ Loaded: {race_session.event['EventName']}")

    except RateLimitExceededError:
        print("🛑 Rate limit hit — stopping early. Re-run in an hour to continue.")
        raise
    except Exception as e:
        print(f"  ⚠️  Skipping 2026 R{round_number}: {e}")
        continue

if not results_2026:
    raise RuntimeError(
        "No 2026 results loaded. Cannot build predictions without current-season data."
    )

df_2026 = pd.concat(results_2026, ignore_index=True)
df_2026['Position']       = pd.to_numeric(df_2026['Position'],       errors='coerce')
df_2026['GridPosition']   = pd.to_numeric(df_2026['GridPosition'],   errors='coerce')
df_2026['QualiPosition']  = pd.to_numeric(df_2026['QualiPosition'],  errors='coerce')
df_2026['SprintPosition'] = pd.to_numeric(df_2026['SprintPosition'], errors='coerce')

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────

# Combine all data for encoding (ensures consistent label encoding)
df_all = pd.concat([df, df_2026], ignore_index=True)
df_all = df_all.sort_values(['FullName', 'Year', 'RoundNumber']).reset_index(drop=True)

# Label encode driver and team
le_driver = LabelEncoder()
le_team = LabelEncoder()
df_all['DriverEncoded'] = le_driver.fit_transform(df_all['FullName'])
df_all['TeamEncoded'] = le_team.fit_transform(df_all['TeamName'])

# Rolling recent form: avg finish position over last 3 races per driver
df_all['RecentForm'] = (
    df_all.groupby('FullName')['Position']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Austria circuit flag
df_all['IsAustria'] = df_all['EventName'].str.contains('Austria', case=False).astype(int)

# Convert types
df_all['QualiPosition']  = pd.to_numeric(df_all['QualiPosition'],  errors='coerce')
df_all['SprintPosition'] = pd.to_numeric(df_all['SprintPosition'], errors='coerce')

# Recent qualifying form (pure pace signal)
df_all['RecentQualiForm'] = (
    df_all.groupby('FullName')['QualiPosition']
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Race craft: avg positions gained/lost from grid → finish over last 5 races
df_all['RaceCraft'] = (
    df_all.groupby('FullName', group_keys=False)
          .apply(lambda g: (g['GridPosition'] - g['Position'])
                           .shift(1).rolling(5, min_periods=1).mean())
)

# Sprint form (NaN for non-sprint weekends — that's fine, RF handles it)
df_all['RecentSprintForm'] = (
    df_all.groupby('FullName')['SprintPosition']
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
# ── 4. TRAIN THE MODEL ────────────────────────────────────────────────────────
print("\n🤖 Training Random Forest model...")

FEATURES = [
    'GridPosition', 'QualiPosition', 'DriverEncoded', 'TeamEncoded',
    'RecentForm', 'RecentQualiForm', 'RecentSprintForm', 'RaceCraft',
    'IsAustria',
]

# Train only on historical data (not 2026)
train_df = df_all[df_all['Year'] < 2026].dropna(subset=FEATURES + ['Position'])
X = train_df[FEATURES]
y = train_df['Position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"  Model MAE: ±{mae:.2f} positions on test data")

# Feature importance
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Feature importance:")
for feat, imp in importance.items():
    print(f"    {feat}: {imp:.3f}")

# ── 5. PREDICT Austria 2026 ─────────────────────────────────────────────────────
print("\n🏁 Building Austria 2026 predictions...")

# Per-driver 2026 stats: averages across the season so far
driver_2026_stats = df_2026.groupby(['FullName', 'TeamName']).agg(
    AvgGrid=('GridPosition', 'mean'),
    AvgQuali=('QualiPosition', 'mean'),
    AvgSprint=('SprintPosition', 'mean'),
    RecentForm=('Position', 'mean'),
    AvgPositionsGained=('Position', lambda s: (
        df_2026.loc[s.index, 'GridPosition'] - s
    ).mean()),
).reset_index()

# Build prediction input
austria_input = driver_2026_stats.copy()
austria_input['IsMonaco'] = 1
austria_input['GridPosition']     = austria_input['AvgGrid']
austria_input['QualiPosition']    = austria_input['AvgQuali']
austria_input['RecentQualiForm']  = austria_input['AvgQuali']
austria_input['RecentSprintForm'] = austria_input['AvgSprint']
austria_input['RaceCraft']        = austria_input['AvgPositionsGained']

# Safely encode — handle any new drivers not seen in training
def safe_encode(encoder, values):
    known = set(encoder.classes_)
    return np.array([
        encoder.transform([v])[0] if v in known else len(encoder.classes_)
        for v in values
    ])

austria_input['DriverEncoded'] = safe_encode(le_driver, austria_input['FullName'])
austria_input['TeamEncoded']   = safe_encode(le_team,   austria_input['TeamName'])
austria_input['IsAustria'] = 1
X_austria = austria_input[FEATURES].fillna(austria_input[FEATURES].mean(numeric_only=True))
austria_input['PredictedPosition'] = model.predict(X_austria)
austria_input = austria_input.sort_values('PredictedPosition').reset_index(drop=True)
austria_input.index += 1  # 1-based ranking

# ── 6. OUTPUT ─────────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("🏆  AUSTRIA GRAND PRIX 2026 — PREDICTED FINISHING ORDER")
print("═" * 55)
print(f"{'Pos':<5} {'Driver':<25} {'Team':<25}")
print("─" * 55)
for pos, row in austria_input.iterrows():
    print(f"{pos:<5} {row['FullName']:<25} {row['TeamName']:<25}")
print("═" * 55)
print(f"\n⚠️  Note: Predictions are based on {len(df)} historical race results")
print("   and 2026 form. Grid positions are estimated from season averages.")
print("   Treat as indicative, not definitive!")