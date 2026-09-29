import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.core.similarity.similarity_engine import SimilarityEngine

def main():
    datasets = ['breast_cancer', 'wine', 'diabetes']
    contexts = {}
    
    for ds in datasets:
        with open(f'results/phase3/{ds}_profile.json', 'r') as f:
            data = json.load(f)
            contexts[ds] = data['numeric_context_vector']
            
    engine = SimilarityEngine()
    
    # Generate diagnostic matrix
    matrix, _ = engine.similarity_matrix(contexts)
    
    print("\n--- SIMILARITY MATRIX ---")
    df_mat = pd.DataFrame(matrix)
    print(df_mat)
    
    # Save matrix
    os.makedirs('results/phase5', exist_ok=True)
    df_mat.to_csv('results/phase5/similarity_matrix.csv')
    
    # Perform LODO
    print("\n--- LODO RETRIEVAL ---")
    lodo_results = {}
    for target in datasets:
        print(f"\nTarget: {target}")
        ranks = engine.rank(target, contexts[target], contexts)
        lodo_results[target] = ranks
        for r in ranks:
            print(f"  {r['rank']}. {r['dataset_id']} - similarity {r['similarity']:.4f} (dist {r['distance']:.4f})")
            
    with open('results/phase5/lodo_retrieval.json', 'w') as f:
        json.dump(lodo_results, f, indent=4)

if __name__ == '__main__':
    main()
