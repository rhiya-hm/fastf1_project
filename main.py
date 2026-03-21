import fastf1
#import os
import pandas as pd

# Optional but recommended: enables local caching
fastf1.Cache.enable_cache('cache')

# Load a session
#session = fastf1.get_session(2025, 'Qatar Grand Prix', 'Q')
#session.load()

# Print some simple information
#print(session.event['EventName'])
#print(session.results[['FullName', 'TeamName', 'Position']].head())

#print all races
schedule = fastf1.get_event_schedule(2025)
#print(schedule[['RoundNumber','EventName','Country']])
schedule = schedule[schedule['EventFormat'] == 'conventional']

winners = []

for round_number in schedule['RoundNumber']:
    session = fastf1.get_session(2023, round_number, 'R')
    session.load()

    winners.append({
        "Race": session.event['EventName'],
        "Winner": session.results.iloc[0]['FullName']
    })

df = pd.DataFrame(winners)
print(df)
