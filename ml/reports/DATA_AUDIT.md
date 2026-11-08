# Local data inventory and quality audit

Generated from the local parquet files. This records current coverage; source availability timing is not inferred from result values.

## Tables

| File | Source | Availability meaning | Rows | Columns | Missing cells | Missing rate |
|---|---|---|---:|---:|---:|---:|
| `raw/races.parquet` | Jolpica calendar | Published calendar; event-level, one row per race | 115 | 11 | 0 | 0.0% |
| `raw/results.parquet` | Jolpica results | Final result/status; known only after the race | 2256 | 18 | 394 | 1.0% |
| `raw/qualifying.parquet` | Jolpica qualifying | Qualifying result; known after qualifying | 2251 | 19 | 3580 | 8.4% |
| `raw/sprints.parquet` | Jolpica sprint | Sprint result; known only after sprint | 552 | 12 | 0 | 0.0% |
| `raw/standings.parquet` | Jolpica standings | Championship after the named round | 2373 | 14 | 13 | 0.0% |
| `raw/pitstops.parquet` | Jolpica pit stops | Observed during/after the race; history only for future events | 3053 | 7 | 0 | 0.0% |
| `raw/weather.parquet` | Open-Meteo archive | Observed weather; not a historical forecast snapshot | 115 | 6 | 12 | 1.7% |
| `canonical/fp_pace.parquet` | OpenF1 adapter | Session lap summaries; local table has no source timestamp | 10260 | 5 | 0 | 0.0% |
| `canonical/starting_grid.parquet` | OpenF1/manual | Grid snapshot; source and as_of may be available | 1 | 7 | 0 | 0.0% |
| `canonical/weather_forecast.parquet` | Open-Meteo forecast | Current forecast snapshot; not a historical archive | 1 | 6 | 0 | 0.0% |
| `raw/openf1_race_sessions.parquet` | OpenF1 sessions/meetings | Historical race session times and circuit metadata | 89 | 12 | 0 | 0.0% |
| `raw/openf1_race_control.parquet` | OpenF1 race_control | Race-control messages with UTC timestamps | 8889 | 14 | 30192 | 24.3% |
| `raw/openf1_safety_car_labels.parquet` | Derived OpenF1 labels | Race-level deployed SC/VSC indicators | 89 | 11 | 0 | 0.0% |
| `processed/driver_race.parquet` | Feature builder | One row per driver × completed race; labels and features | 2256 | 83 | 15155 | 8.1% |

## Row grain and labels

- Calendar races: 115; completed races represented in results: 111; processed races: 111.
- Result rows: 2256; processed driver-race rows: 2256; duplicate processed `(season, round, driver_id)` keys: 0.
- Classified rows under the model rule (`Finished`, `Lapped`, or `+N Lap`): 1916; non-classified proxy rows: 340.
- Retirement DNF labels: 311 positive; 29 excluded DNS/DSQ/withdrawal labels.
- Race-calendar rows missing latitude or longitude: 0.

Raw status is retained. The retirement DNF label treats classified finishers as 0 and retirements/non-classifications as 1; DNS, DSQ, and withdrawals are excluded as unknown/other outcomes. It is not a cause-specific mechanical failure label.

## Provenance limits

- The race calendar has circuit IDs, names, country, latitude and longitude, but no corner map, speed mix, elevation or pit-lane time loss.
- Local OpenF1 pace contains only per-driver/session best-lap summaries and no source timestamp, sector splits or speed-trap samples.
- Archived weather is observed weather. The one-row current forecast file is not a historical forecast archive.
- OpenF1 meeting metadata provides a coarse circuit type; it does not expose corner-speed mix, elevation, pit-lane loss, or verified track geometry in the local table.
- OpenF1 race-control records provide deployed SC/VSC event labels from 2023 onward; the event labels are useful for a cautious race-level baseline, not a deployment-time model.
- Archived weather is observed weather, not a historical forecast archive; local lap summaries are not timestamped and lack sectors/speed traps.
- Therefore high/low-speed car traits and weather-scenario training remain deferred; source-backed circuit-type and prior SC/VSC frequency can be tested.
