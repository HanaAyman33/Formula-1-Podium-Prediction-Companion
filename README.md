# Formula 1 Podium Prediction Companion
 
ACL course project (GUC). The goal is to predict whether a driver finishes on the podium (top 3), using information available **after qualifying and before the race**.
 
## Dataset
[Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020): 14 tables (races, results, qualifying, lap times, pit stops, standings, drivers, constructors, circuits, ...).
 
**Target:** `podium = positionOrder <= 3`, one row per (raceId, driverId).
 
## Milestone 1 – Data Cleaning ✅
- Converted placeholders (`\N`, empty) to NaN and fixed data types in all 14 tables.
- Checked primary and foreign keys and removed duplicate records.
- Removed the Indianapolis 500 races, non-starters and duplicate driver–race rows.
- Fixed invalid values (an impossible Q1 time, wrong pit-stop numbers).
- Labelled missing values (structural / unknown) and every column's leakage status.
- Compared the data before and after cleaning (EDA) and saved the clean tables as parquet.
| | Before | After |
|---|---|---|
| Result rows | 26,759 | 24,566 |
| Races | 1,125 | 1,114 |
 
