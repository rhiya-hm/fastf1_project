"""
tyre_strategy.py
────────────────
Full 2025 season tyre strategy analysis.

Produces three dark-themed charts saved to OUTPUT_DIR:
  01_strategy_timeline_Monaco_2025.png  — per-driver stint map for Monaco 2025
  02_compound_degradation.png — lap time delta vs tyre age per compound
  03_pit_timing.png           — pit stop timing vs race outcome across the season

Usage:  python tyre_strategy.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import fastf1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

fastf1.Cache.enable_cache('../Cache')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
YEAR           = 2025
SHOWCASE_RACE  = 'British Grand Prix'   # race used for the strategy timeline chart
OUTPUT_DIR     = ''

BG = '#0d0d1a'
COMPOUND_COLORS = {
    'SOFT':         '#E8002D',
    'MEDIUM':       '#FFF200',
    'HARD':         '#EBEBEB',
    'INTERMEDIATE': '#39B54A',
    'WET':          '#0067FF',
    'UNKNOWN':      '#888888',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────
def style_ax(ax):
    """Apply consistent dark styling to an axes."""
    ax.set_facecolor(BG)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(True, alpha=0.12, color='white', linestyle='--')


def compound_legend(ax):
    patches = [
        mpatches.Patch(color=COMPOUND_COLORS[c], label=c.capitalize(),
                       edgecolor='#555' if c == 'HARD' else 'none')
        for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    ]
    ax.legend(handles=patches, fontsize=9, facecolor='#1a1a2e',
              labelcolor='white', edgecolor='#444', loc='lower right')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FULL SEASON LAP DATA
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*58}")
print(f"  {YEAR} Full Season — Tyre Strategy Analysis")
print(f"{'─'*58}")
print("📦  Loading full season lap data (uses cache after first run)…\n")

schedule = fastf1.get_event_schedule(YEAR)
schedule = schedule[schedule['EventFormat'] == 'conventional']

all_laps = []
showcase_found = False

for _, event in schedule.iterrows():
    rn = event['RoundNumber']
    en = event['EventName']
    try:
        sess = fastf1.get_session(YEAR, rn, 'R')
        sess.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = sess.laps[[
            'Driver', 'LapNumber', 'LapTime', 'Compound',
            'TyreLife', 'PitInTime', 'PitOutTime', 'IsAccurate'
        ]].copy()

        laps['EventName']   = en
        laps['RoundNumber'] = rn

        # Merge finishing position and full name from results
        res = sess.results[['Abbreviation', 'TeamName', 'Position', 'FullName']].copy()
        res = res.rename(columns={'Abbreviation': 'Driver'})
        laps = laps.merge(res, on='Driver', how='left')

        all_laps.append(laps)
        print(f"  ✅  {en}")

        if en == SHOWCASE_RACE:
            showcase_found = True

    except Exception as e:
        print(f"  ⚠️   Skipping {en}: {e}")
        continue

if not all_laps:
    raise RuntimeError("No race data could be loaded. Check your FastF1 cache / internet connection.")

df = pd.concat(all_laps, ignore_index=True)
df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
df['Compound']       = df['Compound'].fillna('UNKNOWN').str.upper()
df['TyreLife']       = pd.to_numeric(df['TyreLife'], errors='coerce')
df['Position']       = pd.to_numeric(df['Position'], errors='coerce')
df['LapNumber']      = pd.to_numeric(df['LapNumber'], errors='coerce')

if not showcase_found:
    # Fall back to the first race in the dataset
    SHOWCASE_RACE = df['EventName'].iloc[0]
    print(f"\n⚠️   '{SHOWCASE_RACE}' not found — falling back to: {SHOWCASE_RACE}")

print(f"\n  Total laps loaded: {len(df):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHART 1 — STRATEGY TIMELINE (showcase race)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n📊  [1/3]  Building strategy timeline for {SHOWCASE_RACE}…")

race_df = df[df['EventName'] == SHOWCASE_RACE].copy()

# Sort drivers by finishing position
pos_order = (
    race_df.drop_duplicates('Driver')
           .sort_values('Position')
           .reset_index(drop=True)
)
drivers_ordered = pos_order['Driver'].tolist()


def get_stints(driver_laps):
    """Return a DataFrame of stints (compound, start lap, end lap)."""
    dl = driver_laps.sort_values('LapNumber').copy()
    dl['StintNum'] = (dl['Compound'] != dl['Compound'].shift()).cumsum()
    return dl.groupby('StintNum').agg(
        Compound=('Compound', 'first'),
        Start=('LapNumber', 'min'),
        End=('LapNumber', 'max'),
    ).assign(Length=lambda x: x['End'] - x['Start'] + 1).reset_index()


total_laps = int(race_df['LapNumber'].max())

fig1, ax1 = plt.subplots(figsize=(16, max(8, len(drivers_ordered) * 0.55 + 1)))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)

for i, driver in enumerate(drivers_ordered):
    drv_laps = race_df[race_df['Driver'] == driver]
    if drv_laps.empty:
        continue
    stints = get_stints(drv_laps)

    for _, stint in stints.iterrows():
        color = COMPOUND_COLORS.get(stint['Compound'], COMPOUND_COLORS['UNKNOWN'])
        ax1.barh(
            i,
            stint['Length'],
            left=stint['Start'] - 1,
            color=color,
            edgecolor='#1a1a2e',
            linewidth=0.6,
            height=0.72
        )
        if stint['Length'] >= 5:
            text_color = '#111111' if stint['Compound'] != 'HARD' else '#555555'
            ax1.text(
                stint['Start'] - 1 + stint['Length'] / 2,
                i,
                stint['Compound'][0],
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color=text_color
            )

# Y-axis labels: P1 VER, P2 NOR, …
y_labels = []
for i, row in pos_order.iterrows():
    pos  = int(row['Position']) if not pd.isna(row['Position']) else '?'
    name = row.get('FullName', row['Driver'])
    short = name.split()[-1].upper() if isinstance(name, str) else row['Driver']
    y_labels.append(f"P{pos}  {row['Driver']}  {short}")

ax1.set_yticks(range(len(drivers_ordered)))
ax1.set_yticklabels(y_labels, color='white', fontsize=8.5)
ax1.set_xlabel('Lap Number', color='white', fontsize=11)
ax1.set_xlim(0, total_laps + 1)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#333')
ax1.grid(True, alpha=0.1, axis='x', color='white', linestyle='--')
ax1.set_title(
    f'{YEAR} {SHOWCASE_RACE} — Tyre Strategy Per Driver',
    color='white', fontsize=13, fontweight='bold', pad=12
)
compound_legend(ax1)

race_slug = SHOWCASE_RACE.replace(' ', '_').replace('Grand_Prix', 'GP')
out1 = f"{OUTPUT_DIR}/01_strategy_timeline_{race_slug}_{YEAR}.png"
plt.tight_layout()
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"  ✅  Saved → {out1}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHART 2 — COMPOUND DEGRADATION (full season)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n📊  [2/3]  Building compound degradation curves…")

# Only accurate, dry-compound laps with valid tyre age
if 'IsAccurate' in df.columns:
    acc_mask = df['IsAccurate'] == True
else:
    acc_mask = pd.Series(True, index=df.index)

deg_df = df[
    acc_mask &
    df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD']) &
    df['TyreLife'].notna() &
    df['LapTimeSeconds'].notna() &
    (df['TyreLife'] > 0) &
    (df['TyreLife'] <= 50)
].copy()

# Remove outliers: keep laps within 107% of each race/compound median
deg_df['CompoundMedian'] = deg_df.groupby(['EventName', 'Compound'])['LapTimeSeconds'].transform('median')
deg_df = deg_df[deg_df['LapTimeSeconds'] <= deg_df['CompoundMedian'] * 1.07]

# Aggregate: mean lap time by compound × tyre age
deg_stats = (
    deg_df
    .groupby(['Compound', 'TyreLife'])['LapTimeSeconds']
    .agg(mean='mean', std='std', count='count')
    .reset_index()
)
deg_stats = deg_stats[deg_stats['count'] >= 5]  # need enough laps for reliability

fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor(BG)
style_ax(ax2)

for compound in ['SOFT', 'MEDIUM', 'HARD']:
    cdata = deg_stats[deg_stats['Compound'] == compound].sort_values('TyreLife')
    if cdata.empty:
        continue
    color = COMPOUND_COLORS[compound]

    # Express as delta from the compound's own lap-1 baseline (relative degradation)
    baseline = cdata.loc[cdata['TyreLife'].idxmin(), 'mean']
    delta     = cdata['mean'] - baseline

    ax2.plot(
        cdata['TyreLife'], delta,
        color=color, linewidth=2.5, label=compound.capitalize(), zorder=3
    )
    ax2.fill_between(
        cdata['TyreLife'],
        delta - cdata['std'].fillna(0),
        delta + cdata['std'].fillna(0),
        color=color, alpha=0.10, zorder=1
    )

ax2.axhline(0, color='#666', linewidth=0.9, linestyle='--', zorder=2)
ax2.set_xlabel('Tyre Age (laps)', color='white', fontsize=11)
ax2.set_ylabel('Lap Time vs Fresh Tyre (seconds)', color='white', fontsize=11)
ax2.set_title(
    f'{YEAR} Season — Tyre Degradation by Compound\n'
    'Aggregated across all dry races  ·  shaded band = ±1 std dev',
    color='white', fontsize=12, fontweight='bold', pad=10
)
ax2.legend(fontsize=11, facecolor='#1a1a2e', labelcolor='white', edgecolor='#444')

out2 = f"{OUTPUT_DIR}/02_compound_degradation.png"
plt.tight_layout()
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"  ✅  Saved → {out2}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHART 3 — PIT STOP TIMING (full season)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n📊  [3/3]  Building pit stop timing analysis…")

# Pit laps = laps where PitInTime is not NaT
pit_laps = df[df['PitInTime'].notna()].copy()
pit_laps = pit_laps.dropna(subset=['Position', 'LapNumber'])

# First pit stop per driver per race
first_pits = (
    pit_laps
    .groupby(['Driver', 'EventName'])
    .agg(
        FirstPitLap=('LapNumber', 'min'),
        Position=('Position', 'first'),
        TeamName=('TeamName', 'first'),
        RoundNumber=('RoundNumber', 'first')
    )
    .reset_index()
)

# Normalise pit lap as % of race distance
race_lengths = df.groupby('EventName')['LapNumber'].max().rename('TotalLaps')
first_pits   = first_pits.merge(race_lengths, on='EventName')
first_pits['PitPct'] = (first_pits['FirstPitLap'] / first_pits['TotalLaps'] * 100).round(1)
first_pits   = first_pits.dropna(subset=['PitPct', 'Position'])

# Pit stop count per driver per race
pit_counts = (
    pit_laps
    .groupby(['Driver', 'EventName'])
    .size()
    .reset_index(name='Stops')
)
pit_counts = pit_counts.merge(
    df[['Driver', 'EventName', 'Position']].drop_duplicates(['Driver', 'EventName']),
    on=['Driver', 'EventName']
)
pit_counts['Position'] = pd.to_numeric(pit_counts['Position'], errors='coerce')

# Average finishing position by number of stops
stop_avg = (
    pit_counts
    .groupby('Stops')['Position']
    .agg(mean='mean', std='std', count='count')
    .reset_index()
)
stop_avg = stop_avg[(stop_avg['count'] >= 5) & (stop_avg['Stops'] <= 4)]

# Team colour map
teams       = sorted(first_pits['TeamName'].dropna().unique())
team_cmap   = plt.cm.tab20(np.linspace(0, 1, max(len(teams), 1)))
team_colors = {t: team_cmap[i] for i, t in enumerate(teams)}

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 8))
fig3.patch.set_facecolor(BG)

# ── LEFT: scatter — first pit % vs finishing position ─────────────────────────
style_ax(ax3a)
for team in teams:
    td = first_pits[first_pits['TeamName'] == team]
    ax3a.scatter(
        td['PitPct'], td['Position'],
        color=team_colors[team], s=45, alpha=0.72,
        label=team, zorder=3
    )

# Trend line
z = np.polyfit(first_pits['PitPct'], first_pits['Position'], 1)
x_line = np.linspace(first_pits['PitPct'].min(), first_pits['PitPct'].max(), 100)
ax3a.plot(x_line, np.poly1d(z)(x_line),
          color='white', linewidth=1.5, linestyle='--', alpha=0.45, zorder=2)

ax3a.invert_yaxis()
ax3a.set_xlabel('First Pit Stop (% into race)', color='white', fontsize=11)
ax3a.set_ylabel('Final Race Position', color='white', fontsize=11)
ax3a.set_title(
    'First Pit Timing vs Final Position\n(all races, all drivers)',
    color='white', fontsize=11, fontweight='bold'
)
ax3a.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
            edgecolor='#444', ncol=2, loc='lower right',
            markerscale=1.2)

# ── RIGHT: bar chart — avg finish by number of stops ─────────────────────────
style_ax(ax3b)
bar_colors = [COMPOUND_COLORS['SOFT'], COMPOUND_COLORS['MEDIUM'],
              COMPOUND_COLORS['HARD'], COMPOUND_COLORS['INTERMEDIATE']]

bars = ax3b.bar(
    stop_avg['Stops'].astype(str).values,
    stop_avg['mean'].values,
    color=bar_colors[:len(stop_avg)],
    edgecolor='#1a1a2e',
    linewidth=0.8,
    width=0.55
)
ax3b.errorbar(
    x=range(len(stop_avg)),
    y=stop_avg['mean'].values,
    yerr=stop_avg['std'].fillna(0).values,
    fmt='none', color='white', capsize=5, linewidth=1.2, alpha=0.6, zorder=5
)
for bar, (_, row) in zip(bars, stop_avg.iterrows()):
    ax3b.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"n={int(row['count'])}",
        ha='center', va='bottom', color='white', fontsize=9
    )

ax3b.invert_yaxis()
ax3b.set_xlabel('Number of Pit Stops', color='white', fontsize=11)
ax3b.set_ylabel('Avg Finishing Position', color='white', fontsize=11)
ax3b.set_title(
    f'{YEAR} Season — Average Finish Position by Pit Stop Count\n'
    '(error bars = ±1 std dev)',
    color='white', fontsize=11, fontweight='bold'
)

fig3.suptitle(
    f'{YEAR} Season — Pit Stop Strategy Analysis',
    color='white', fontsize=14, fontweight='bold', y=1.01
)

out3 = f"{OUTPUT_DIR}/03_pit_timing.png"
plt.tight_layout()
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"  ✅  Saved → {out3}")
plt.close()

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*58}")
print(f"  🏁  All charts saved to '{OUTPUT_DIR}/'")
print(f"      {os.path.basename(out1)}   — {SHOWCASE_RACE}")
print(f"      02_compound_degradation.png — full season tyre curves")
print(f"      03_pit_timing.png           — pit strategy vs outcome")
print(f"{'═'*58}\n")
