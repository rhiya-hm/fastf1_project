"""
telemetry_compare.py
────────────────────
Qualifying telemetry comparison for two drivers at a given race.

Produces three visualisations on a dark-themed figure:
  1. Individual track maps coloured by each driver's speed
  2. A delta track map — green = Driver 1 faster, red = Driver 2 faster
  3. Speed trace overlay with shaded advantage areas

Usage:  python telemetry_compare.py
Output: telemetry_compare_output.png  (also opens in a window)
"""

import warnings

from matplotlib.colorbar import Colorbar

warnings.filterwarnings("ignore")

from fastf1 import plotting, get_session, Cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
#import matplotlib.patches as patches

# ── CACHE & CONFIG ─────────────────────────────────────────────────────────────
Cache.enable_cache('Cache')

YEAR        = 2026
EVENT       = 'Miami Grand Prix'
EVENT       = 'Miami Grand Prix'
SESSION_TYPE = 'Q'
# Q = Qualifying
# R = Race
OUTPUT_FILE = f'telemetry_compare_files/{YEAR}_{EVENT.replace(" ", "_")}_telemetry_{SESSION_TYPE}.png'

# Fallback colours if FastF1 can't provide team colours
FALLBACK_COLORS = ['#FF1E00', '#0067FF']

# ── HELPERS ────────────────────────────────────────────────────────────────────
def fmt_lap(td):
    """Format a timedelta lap time as M:SS.mmm"""
    total = td.total_seconds()
    m = int(total // 60)
    s = total % 60
    return f"{m}:{s:06.3f}"


def get_driver_color(driver_abbr, session, fallback):
    """Safely retrieve driver colour from FastF1, falling back gracefully."""
    try:
        return plotting.get_driver_color(driver_abbr, session=session)
    except Exception:
        pass
    try:
        return plotting.DRIVER_COLORS.get(driver_abbr, fallback)
    except Exception:
        return fallback


def resample_telemetry(tel, dist_axis):
    """Interpolate telemetry channels onto a common distance axis."""
    speed    = np.interp(dist_axis, tel['Distance'], tel['Speed'])
    throttle = np.interp(dist_axis, tel['Distance'], tel['Throttle'])
    brake    = np.interp(dist_axis, tel['Distance'], tel['Brake'].astype(float))
    x        = np.interp(dist_axis, tel['Distance'], tel['X'])
    y        = np.interp(dist_axis, tel['Distance'], tel['Y'])
    return speed, throttle, brake, x, y


def make_track_map(ax, x, y, values, cmap, norm, title, bg='#0d0d1a'):
    """Draw a track map with segments coloured by `values`."""
    ax.set_facecolor(bg)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=10, pad=6, fontweight='bold')

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=3.5, zorder=2, capstyle='round')
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def add_colorbar(ax, cmap, norm, label, bg='#0d0d1a'):
    """Add a vertical colorbar inside a dedicated axes."""
    ax.set_facecolor(bg)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar: Colorbar = plt.colorbar(sm, ax=ax, fraction=0.35, pad=0.05,
                        orientation='vertical')
    cbar.set_label(label, color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor


# ── LOAD SESSION ───────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  {YEAR} {EVENT} — Qualifying Telemetry Comparison")
print(f"{'─'*55}")
print(f"📦  Loading session data…")

session = get_session(YEAR, EVENT, SESSION_TYPE)
session.load(laps=True, telemetry=True, weather=False, messages=False)

# ── PICK DRIVERS ───────────────────────────────────────────────────────────────
results = session.results.sort_values('Position').reset_index(drop=True)
d1_row, d2_row = results.iloc[0], results.iloc[1]

driver1, driver2 = d1_row['Abbreviation'], d2_row['Abbreviation']
team1,   team2   = d1_row['TeamName'],     d2_row['TeamName']
color1  = get_driver_color(driver1, session, FALLBACK_COLORS[0])
color2  = get_driver_color(driver2, session, FALLBACK_COLORS[1])

print(f"  P1 → {driver1:>3}  ({team1})")
print(f"  P2 → {driver2:>3}  ({team2})")

# ── FASTEST LAPS ───────────────────────────────────────────────────────────────
lap1 = session.laps.pick_driver(driver1).pick_fastest()
lap2 = session.laps.pick_driver(driver2).pick_fastest()

t1 = lap1.get_telemetry()
t2 = lap2.get_telemetry()

lap1_str = fmt_lap(lap1['LapTime'])
lap2_str = fmt_lap(lap2['LapTime'])
delta_s  = lap1['LapTime'].total_seconds() - lap2['LapTime'].total_seconds()

print(f"\n  {driver1} lap time : {lap1_str}")
print(f"  {driver2} lap time : {lap2_str}")
print(f"  Delta          : {delta_s:+.3f}s  ({driver1} vs {driver2})")

# ── RESAMPLE ONTO COMMON DISTANCE ──────────────────────────────────────────────
NUM_POINTS = 1500
d_min = max(t1['Distance'].min(), t2['Distance'].min())
d_max = min(t1['Distance'].max(), t2['Distance'].max())
dist  = np.linspace(d_min, d_max, NUM_POINTS)

speed1, throttle1, brake1, x1, y1 = resample_telemetry(t1, dist)
speed2, throttle2, brake2, x2, y2 = resample_telemetry(t2, dist)

# Use driver 1's GPS trace as the reference track path
x_ref, y_ref = x1, y1

speed_delta = speed1 - speed2   # positive ⟹ driver 1 faster at that point

# ── FIGURE LAYOUT ──────────────────────────────────────────────────────────────
BG = '#0d0d1a'
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor(BG)

gs = matplotlib.gridspec.GridSpec(
    3, 4,
    figure=fig,
    height_ratios=[1, 1, 0.9],
    width_ratios=[1, 1, 1, 0.08],
    hspace=0.45,
    wspace=0.25
)

# ── ROW 0: individual speed maps ───────────────────────────────────────────────
speed_global_norm = Normalize(
    vmin=min(speed1.min(), speed2.min()),
    vmax=max(speed1.max(), speed2.max())
)

ax_d1   = fig.add_subplot(gs[0, 0:2])
ax_d2   = fig.add_subplot(gs[0, 2])
ax_cb1  = fig.add_subplot(gs[0, 3])

make_track_map(ax_d1, x_ref, y_ref, speed1, 'plasma', speed_global_norm,
               f'{driver1}  ·  {lap1_str}')
make_track_map(ax_d2, x_ref, y_ref, speed2, 'plasma', speed_global_norm,
               f'{driver2}  ·  {lap2_str}')
add_colorbar(ax_cb1, 'plasma', speed_global_norm, 'Speed (km/h)')

# ── ROW 1: speed-delta track map ───────────────────────────────────────────────
max_abs_delta  = float(np.percentile(np.abs(speed_delta), 95)) or 1.0
delta_norm     = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0, vmax=max_abs_delta)

ax_delta  = fig.add_subplot(gs[1, 0:3])
ax_cb2    = fig.add_subplot(gs[1, 3])

make_track_map(
    ax_delta, x_ref, y_ref, speed_delta, 'RdYlGn', delta_norm,
    f'Speed Advantage Map  ·  '
    f'Green = {driver1} faster  ·  Red = {driver2} faster'
)

# Annotate lap delta on the delta map
sign   = "+" if delta_s > 0 else ""
note   = f"Lap Δ  {sign}{delta_s:.3f}s"
ax_delta.text(
    0.02, 0.05, note,
    transform=ax_delta.transAxes,
    color='white', fontsize=11, fontweight='bold',
    bbox=dict(facecolor='#1a1a2e', edgecolor='#555', boxstyle='round,pad=0.4')
)

add_colorbar(ax_cb2, 'RdYlGn', delta_norm, 'Speed Δ (km/h)')

# ── ROW 2: speed trace ─────────────────────────────────────────────────────────
ax_speed = fig.add_subplot(gs[2, 0:4])
ax_speed.set_facecolor('#0d0d1a')

ax_speed.plot(dist / 1000, speed1, color=color1, linewidth=1.8,
              label=f'{driver1}  ({lap1_str})', zorder=3)
ax_speed.plot(dist / 1000, speed2, color=color2, linewidth=1.8,
              label=f'{driver2}  ({lap2_str})', zorder=3, linestyle='--', alpha=0.9)

# Shade advantage regions
ax_speed.fill_between(
    dist / 1000, speed1, speed2,
    where=(speed1 > speed2), alpha=0.18, color=color1, zorder=1
)
ax_speed.fill_between(
    dist / 1000, speed1, speed2,
    where=(speed2 > speed1), alpha=0.18, color=color2, zorder=1
)

ax_speed.set_xlabel('Distance (km)', color='white', fontsize=11)
ax_speed.set_ylabel('Speed (km/h)', color='white', fontsize=11)
ax_speed.set_title('Speed Trace', color='white', fontsize=12, fontweight='bold')
ax_speed.tick_params(colors='white', labelsize=9)
for spine in ax_speed.spines.values():
    spine.set_edgecolor('#333')
ax_speed.legend(
    fontsize=10, facecolor='#1a1a2e', labelcolor='white',
    edgecolor='#444', loc='upper right'
)
ax_speed.grid(True, alpha=0.15, color='white', linestyle='--')
ax_speed.set_xlim(dist.min() / 1000, dist.max() / 1000)

# ── MAIN TITLE ─────────────────────────────────────────────────────────────────
fig.suptitle(
    f'{YEAR}  {EVENT}  ·  Qualifying Telemetry  ·  {driver1} vs {driver2}',
    color='white', fontsize=15, fontweight='bold', y=1.01
)

# ── SAVE & SHOW ────────────────────────────────────────────────────────────────
plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
print(f"\n✅  Chart saved → {OUTPUT_FILE}")
plt.show()
