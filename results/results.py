from typing import Any
from datetime import datetime, timezone
import fastf1
import pandas as pd
from fastf1.core import Session
import matplotlib.pyplot as plt
from pathlib import Path


CACHE_DIR = Path(__file__).resolve().parent.parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

schedule = fastf1.get_event_schedule(2024)
schedule = schedule[schedule['RoundNumber'] > 0]

winners: list[Any] = []

for round_number in schedule['RoundNumber']:
    session: Session = fastf1.get_session(2024, round_number, 'R')
    if datetime.now(timezone.utc) < session.event['Session5DateUtc'].replace(tzinfo=timezone.utc):
        break
    session.load()

    winners.append({
        "Race": session.event['EventName'],
        "Winner": session.results.iloc[0]['FullName']
    })

df = pd.DataFrame(winners)
print(df)


fig, ax = plt.subplots(figsize=(6, len(df) * 0.6 + 1))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)
plt.tight_layout()
plt.savefig('2024_results.png', dpi=150, bbox_inches='tight')