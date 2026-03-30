from datetime import datetime, timezone
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('cache')

# ── 1. COLLECT HISTORICAL RACE DATA (2022–2025) ──────────────────────────────
print("📦 Loading historical race data (2022–2025)... this may take a while")

all_results = []

for year in range(2022, 2026):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] == 'conventional']

    for round_number in schedule['RoundNumber']:
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load(telemetry=False, weather=False, messages=False)

            results = session.results[['FullName', 'TeamName', 'GridPosition', 'Position']].copy()
            results['Year'] = year
            results['RoundNumber'] = round_number
            results['EventName'] = session.event['EventName']
            all_results.append(results)
        except Exception as e:
            print(f"  ⚠️  Skipping {year} Round {round_number}: {e}")
            continue

df = pd.concat(all_results, ignore_index=True)
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df = df.dropna(subset=['Position', 'GridPosition'])

# ── 2. COLLECT 2026 RACE DATA (completed races only) ─────────────────────────
print("\n📦 Loading 2026 race data...")

schedule_2026 = fastf1.get_event_schedule(2026)
schedule_2026 = schedule_2026[schedule_2026['EventFormat'] == 'conventional']

results_2026 = []
for round_number in schedule_2026['RoundNumber']:
    session = fastf1.get_session(2026, round_number, 'R')
    if datetime.now(timezone.utc) < session.event['Session5DateUtc'].replace(tzinfo=timezone.utc):
        break
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results[['FullName', 'TeamName', 'GridPosition', 'Position']].copy()
    results['Year'] = 2026
    results['RoundNumber'] = round_number
    results['EventName'] = session.event['EventName']
    results_2026.append(results)
    print(f"  ✅ Loaded: {session.event['EventName']}")

df_2026 = pd.concat(results_2026, ignore_index=True)
df_2026['Position'] = pd.to_numeric(df_2026['Position'], errors='coerce')
df_2026['GridPosition'] = pd.to_numeric(df_2026['GridPosition'], errors='coerce')

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

# Miami circuit flag
df_all['IsMiami'] = df_all['EventName'].str.contains('Miami', case=False).astype(int)

# ── 4. TRAIN THE MODEL ────────────────────────────────────────────────────────
print("\n🤖 Training Random Forest model...")

FEATURES = ['GridPosition', 'DriverEncoded', 'TeamEncoded', 'RecentForm', 'IsMiami']

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

# ── 5. PREDICT MIAMI 2026 ─────────────────────────────────────────────────────
print("\n🏁 Building Miami 2026 predictions...")

# Per-driver 2026 stats: avg grid position and recent form
driver_2026_stats = df_2026.groupby(['FullName', 'TeamName']).agg(
    AvgGrid=('GridPosition', 'mean'),
    RecentForm=('Position', 'mean')
).reset_index()

# Build prediction input
miami_input = driver_2026_stats.copy()
miami_input['IsMiami'] = 1
miami_input['GridPosition'] = miami_input['AvgGrid']

# Safely encode — handle any new drivers not seen in training
def safe_encode(encoder, values):
    known = set(encoder.classes_)
    return np.array([
        encoder.transform([v])[0] if v in known else len(encoder.classes_)
        for v in values
    ])

miami_input['DriverEncoded'] = safe_encode(le_driver, miami_input['FullName'])
miami_input['TeamEncoded'] = safe_encode(le_team, miami_input['TeamName'])

X_miami = miami_input[FEATURES].fillna(miami_input[FEATURES].mean())
miami_input['PredictedPosition'] = model.predict(X_miami)
miami_input = miami_input.sort_values('PredictedPosition').reset_index(drop=True)
miami_input.index += 1  # 1-based ranking

# ── 6. OUTPUT ─────────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("🏆  MIAMI GRAND PRIX 2026 — PREDICTED FINISHING ORDER")
print("═" * 55)
print(f"{'Pos':<5} {'Driver':<25} {'Team':<25}")
print("─" * 55)
for pos, row in miami_input.iterrows():
    print(f"{pos:<5} {row['FullName']:<25} {row['TeamName']:<25}")
print("═" * 55)
print(f"\n⚠️  Note: Predictions are based on {len(df)} historical race results")
print("   and 2026 form. Grid positions are estimated from season averages.")
print("   Treat as indicative, not definitive!")
