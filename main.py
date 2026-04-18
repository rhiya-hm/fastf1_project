from typing import Any
from datetime import datetime, timezone
import fastf1
import pandas as pd
from fastf1.core import Session

# Optional but recommended: enables local caching
fastf1.Cache.enable_cache('cache')

schedule = fastf1.get_event_schedule(2026)
schedule = schedule[schedule['EventFormat'] == 'conventional']

winners: list[Any] = []

for round_number in schedule['RoundNumber']:
    session: Session = fastf1.get_session(2026, round_number, 'R')
    if datetime.now(timezone.utc) < session.event['Session5DateUtc'].replace(tzinfo=timezone.utc):
        break
    session.load()

    winners.append({
        "Race": session.event['EventName'],
        "Winner": session.results.iloc[0]['FullName']
    })

df = pd.DataFrame(winners)
print(df)
