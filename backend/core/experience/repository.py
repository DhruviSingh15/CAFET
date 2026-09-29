import sqlite3
import json
from typing import Dict, List, Any, Optional

class ExperienceRepository:
    def __init__(self, db_path: str = "cafet_experience.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experience (
                    experience_id TEXT PRIMARY KEY,
                    source_dataset_id TEXT NOT NULL,
                    source_dataset_name TEXT NOT NULL,
                    dataset_context TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    transformation TEXT NOT NULL,
                    source_features TEXT NOT NULL,
                    baseline_score REAL,
                    candidate_score REAL,
                    utility_score REAL,
                    gain REAL,
                    evaluation_time REAL,
                    computational_cost REAL,
                    rank INTEGER,
                    is_successful BOOLEAN,
                    random_seed INTEGER,
                    timestamp TEXT,
                    schema_version TEXT,
                    UNIQUE(source_dataset_id, candidate_id)
                )
            ''')
            conn.commit()

    def add_experience(self, exp: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experience (
                    experience_id, source_dataset_id, source_dataset_name, dataset_context,
                    candidate_id, transformation, source_features, baseline_score,
                    candidate_score, utility_score, gain, evaluation_time, computational_cost,
                    rank, is_successful, random_seed, timestamp, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exp['experience_id'], exp['source_dataset_id'], exp['source_dataset_name'],
                json.dumps(exp['dataset_context']), exp['candidate_id'], exp['transformation'],
                json.dumps(exp['source_features']), exp['baseline_score'], exp['candidate_score'],
                exp['utility_score'], exp['gain'], exp['evaluation_time'], exp['computational_cost'],
                exp['rank'], exp['is_successful'], exp['random_seed'], exp['timestamp'], exp['schema_version']
            ))
            conn.commit()

    def get_experience(self, experience_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM experience WHERE experience_id = ?', (experience_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def list_experiences(self, exclude_dataset_id: str = None, source_dataset_id: str = None, 
                         candidate_id: str = None, transformation: str = None, 
                         is_successful: bool = None) -> List[Dict[str, Any]]:
        query = 'SELECT * FROM experience WHERE 1=1'
        params = []
        
        if exclude_dataset_id:
            query += ' AND source_dataset_id != ?'
            params.append(exclude_dataset_id)
        if source_dataset_id:
            query += ' AND source_dataset_id = ?'
            params.append(source_dataset_id)
        if candidate_id:
            query += ' AND candidate_id = ?'
            params.append(candidate_id)
        if transformation:
            query += ' AND transformation = ?'
            params.append(transformation)
        if is_successful is not None:
            query += ' AND is_successful = ?'
            params.append(1 if is_successful else 0)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM experience')
            return cursor.fetchone()[0]
            
    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM experience')
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d['dataset_context'] = json.loads(d['dataset_context'])
        d['source_features'] = json.loads(d['source_features'])
        d['is_successful'] = bool(d['is_successful'])
        return d
