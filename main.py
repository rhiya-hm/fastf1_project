from typing import Any
from datetime import datetime, timezone
import fastf1
#import os
import pandas as pd
from fastf1.core import Session

# Optional but recommended: enables local caching
fastf1.Cache.enable_cache('cache')

# Load a session
#session = fastf1.get_session(2025, 'Qatar Grand Prix', 'Q')
#session.load()

# Print some simple information
#print(session.event['EventName'])
#print(session.results[['FullName', 'TeamName', 'Position']].head())

#print all races
schedule = fastf1.get_event_schedule(2026)
#print(schedule[['RoundNumber','EventName','Country']])
schedule = schedule[schedule['EventFormat'] == 'conventional']

winners: list[Any] = []

round_number: object
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
