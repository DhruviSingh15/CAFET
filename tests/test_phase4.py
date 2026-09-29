import pytest
import os
import json
import uuid
import datetime
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from backend.core.experience.repository import ExperienceRepository

@pytest.fixture
def repo(tmp_path):
    db_path = tmp_path / "test.db"
    return ExperienceRepository(str(db_path))

def test_insert_and_read(repo):
    exp = {
        'experience_id': str(uuid.uuid4()),
        'source_dataset_id': 'test_ds',
        'source_dataset_name': 'test_ds',
        'dataset_context': {'n_rows': 100},
        'candidate_id': 'ADD(a,b)',
        'transformation': 'ADD',
        'source_features': ['a', 'b'],
        'baseline_score': 0.8,
        'candidate_score': 0.85,
        'utility_score': 0.05,
        'gain': 0.05,
        'evaluation_time': 0.1,
        'computational_cost': 0.1,
        'rank': 1,
        'is_successful': True,
        'random_seed': 42,
        'timestamp': datetime.datetime.now().isoformat(),
        'schema_version': "1.0"
    }
    repo.add_experience(exp)
    
    fetched = repo.get_experience(exp['experience_id'])
    assert fetched['candidate_id'] == 'ADD(a,b)'
    assert fetched['dataset_context']['n_rows'] == 100
    assert fetched['is_successful'] is True

def test_duplicate_handling(repo):
    exp = {
        'experience_id': '123',
        'source_dataset_id': 'test_ds',
        'source_dataset_name': 'test_ds',
        'dataset_context': {},
        'candidate_id': 'ADD(a,b)',
        'transformation': 'ADD',
        'source_features': ['a', 'b'],
        'baseline_score': 0.8,
        'candidate_score': 0.85,
        'utility_score': 0.05,
        'gain': 0.05,
        'evaluation_time': 0.1,
        'computational_cost': 0.1,
        'rank': 1,
        'is_successful': True,
        'random_seed': 42,
        'timestamp': datetime.datetime.now().isoformat(),
        'schema_version': "1.0"
    }
    repo.add_experience(exp)
    
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_experience(exp)
        
def test_filtering(repo):
    def make_exp(ds, cid, trans, succ):
        return {
            'experience_id': str(uuid.uuid4()),
            'source_dataset_id': ds,
            'source_dataset_name': ds,
            'dataset_context': {},
            'candidate_id': cid,
            'transformation': trans,
            'source_features': [],
            'baseline_score': 0.0,
            'candidate_score': 0.1 if succ else -0.1,
            'utility_score': 0.1 if succ else -0.1,
            'gain': 0.1 if succ else -0.1,
            'evaluation_time': 0.1,
            'computational_cost': 0.1,
            'rank': 1,
            'is_successful': succ,
            'random_seed': 42,
            'timestamp': datetime.datetime.now().isoformat(),
            'schema_version': "1.0"
        }
    
    repo.add_experience(make_exp('ds1', 'ADD(a)', 'ADD', True))
    repo.add_experience(make_exp('ds2', 'ADD(b)', 'ADD', False))
    repo.add_experience(make_exp('ds3', 'LOG(a)', 'LOG', True))
    
    assert repo.count() == 3
    assert len(repo.list_experiences(exclude_dataset_id='ds1')) == 2
    assert len(repo.list_experiences(is_successful=True)) == 2
    assert len(repo.list_experiences(transformation='ADD')) == 2
    assert len(repo.list_experiences(candidate_id='LOG(a)')) == 1

def test_candidate_identity(repo):
    repo.add_experience({
        'experience_id': str(uuid.uuid4()), 'source_dataset_id': 'ds1', 'source_dataset_name': 'ds1',
        'dataset_context': {}, 'candidate_id': 'ADD(a,b)', 'transformation': 'ADD',
        'source_features': ['a', 'b'], 'baseline_score': 0.8, 'candidate_score': 0.85,
        'utility_score': 0.05, 'gain': 0.05, 'evaluation_time': 0.1, 'computational_cost': 0.1,
        'rank': 1, 'is_successful': True, 'random_seed': 42, 'timestamp': '', 'schema_version': "1.0"
    })
    repo.add_experience({
        'experience_id': str(uuid.uuid4()), 'source_dataset_id': 'ds1', 'source_dataset_name': 'ds1',
        'dataset_context': {}, 'candidate_id': 'ADD(a,c)', 'transformation': 'ADD',
        'source_features': ['a', 'c'], 'baseline_score': 0.8, 'candidate_score': 0.85,
        'utility_score': 0.05, 'gain': 0.05, 'evaluation_time': 0.1, 'computational_cost': 0.1,
        'rank': 2, 'is_successful': True, 'random_seed': 42, 'timestamp': '', 'schema_version': "1.0"
    })
    
    assert repo.count() == 2
    c1 = repo.list_experiences(candidate_id='ADD(a,b)')
    assert len(c1) == 1
    c2 = repo.list_experiences(candidate_id='ADD(a,c)')
    assert len(c2) == 1
