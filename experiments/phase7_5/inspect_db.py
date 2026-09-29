import sqlite3
import pandas as pd
import json
import os

db_path = "../../results/phase4/experience.db"
conn = sqlite3.connect(db_path)

print("--- SCHEMA ---")
print(pd.read_sql("PRAGMA table_info(experience)", conn))

print("\n--- SAMPLE ---")
print(pd.read_sql("SELECT * FROM experience LIMIT 2", conn).to_dict(orient="records"))

print("\n--- DATASETS ---")
print(pd.read_sql("SELECT source_dataset_id, COUNT(*) FROM experience GROUP BY source_dataset_id", conn))
