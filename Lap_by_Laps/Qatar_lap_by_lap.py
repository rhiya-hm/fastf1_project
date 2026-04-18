import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fastf1.Cache.enable_cache('cache')

# ── 1. LOAD SESSION ───────────────────────────────────────────────────────────
print("📦 Loading 2025 Qatar GP race data...")
session = fastf1.get_session(2025, 'Qatar Grand Prix', 'R')
session.load(telemetry=False, weather=False, messages=False)

# ── 2. GET LAP DATA ───────────────────────────────────────────────────────────
laps = session.laps.copy()

# Convert lap time to seconds for easier analysis
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Drop laps with no time (in-laps, out-laps, retirements)
laps = laps.dropna(subset=['LapTimeSeconds'])

# Filter out obvious outliers (safety car laps, pit laps etc.)
# Keep only laps within 115% of each driver's personal best
laps['PersonalBest'] = laps.groupby('Driver')['LapTimeSeconds'].transform('min')
laps = laps[laps['LapTimeSeconds'] <= laps['PersonalBest'] * 1.15]

# ── 3. PRINT TABLE ────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("🏁  2025 QATAR GRAND PRIX — LAP TIME SUMMARY PER DRIVER")
print("═" * 70)
print(f"{'Driver':<6} {'Team':<25} {'Best Lap':>10} {'Avg Lap':>10} {'Laps':>6}")
print("─" * 70)

# Get finishing order
results = session.results[['Abbreviation', 'TeamName', 'Position']].copy()
results = results.sort_values('Position')

summary_rows = []
for _, row in results.iterrows():
    drv = row['Abbreviation']
    team = row['TeamName']
    drv_laps = laps[laps['Driver'] == drv]['LapTimeSeconds']

    if drv_laps.empty:
        continue

    best = drv_laps.min()
    avg = drv_laps.mean()

    def fmt(seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:06.3f}"

    print(f"{drv:<6} {team:<25} {fmt(best):>10} {fmt(avg):>10} {len(drv_laps):>6}")
    summary_rows.append({'Driver': drv, 'Team': team, 'BestLap': best, 'AvgLap': avg, 'Laps': len(drv_laps)})

print("═" * 70)
print("* Outlier laps (>115% of personal best) excluded from averages")

# ── 4. CHART — LAP TIMES PER DRIVER OVER RACE ────────────────────────────────
print("\n📊 Generating chart...")

drivers = [r['Driver'] for r in summary_rows]
colors = cm.tab20(np.linspace(0, 1, len(drivers)))

fig, axes = plt.subplots(2, 1, figsize=(16, 14))
fig.suptitle('2025 Qatar Grand Prix — Lap-by-Lap Analysis', fontsize=16, fontweight='bold', y=0.98)

# ── Plot 1: Lap times over race for all drivers ───────────────────────────────
ax1 = axes[0]
for i, drv in enumerate(drivers):
    drv_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
    ax1.plot(
        drv_laps['LapNumber'],
        drv_laps['LapTimeSeconds'],
        label=drv,
        color=colors[i],
        linewidth=1.2,
        alpha=0.85,
        marker='o',
        markersize=2
    )

ax1.set_xlabel('Lap Number', fontsize=11)
ax1.set_ylabel('Lap Time (seconds)', fontsize=11)
ax1.set_title('Lap Times Across the Race (outliers removed)', fontsize=12)
ax1.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.7)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()  # Lower time = faster = better at top

# ── Plot 2: Best lap comparison bar chart ─────────────────────────────────────
ax2 = axes[1]
summary_df = pd.DataFrame(summary_rows).sort_values('BestLap')

bars = ax2.barh(
    summary_df['Driver'],
    summary_df['BestLap'],
    color=[colors[drivers.index(d)] for d in summary_df['Driver']],
    edgecolor='white',
    linewidth=0.5
)

# Add lap time labels on bars
for bar, val in zip(bars, summary_df['BestLap']):
    m = int(val // 60)
    s = val % 60
    ax2.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{m}:{s:06.3f}",
        va='center',
        fontsize=8
    )

ax2.set_xlabel('Best Lap Time (seconds)', fontsize=11)
ax2.set_title('Fastest Lap Per Driver', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(summary_df['BestLap'].min() - 2, summary_df['BestLap'].max() + 4)

plt.tight_layout()
plt.savefig('qatar_2025_lap_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'qatar_2025_lap_analysis.png'")
plt.show()