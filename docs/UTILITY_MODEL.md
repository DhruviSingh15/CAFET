# Candidate-Level Utility Predictor

Estimates: "How useful is this specific candidate transformation for this specific dataset?"

## Inputs
- Dataset context
- Transformation
- Source feature metadata
- Feature interaction metadata
- Historical transfer information
- Task context
- Transformation complexity
- Similarity information

## Outputs
- Predicted utility
- Uncertainty/confidence

## Model Selection
- Start with interpretable tree-based ML: Random Forest, Gradient Boosting, XGBoost/LightGBM.
- Trained ONLY on information available before evaluating the target dataset.
