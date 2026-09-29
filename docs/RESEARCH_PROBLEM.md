# Research Problem: Context-Aware Feature Experience Transfer

## 1. Research Problem
Automated Feature Engineering (AutoFE) often explores a massive search space of candidate feature transformations, making it computationally expensive. Current methods often evaluate candidates uninformedly or rely on static global rules. The core problem is how to efficiently guide the search process for new, unseen datasets by transferring historical feature-engineering experiences acquired from previously processed datasets, constrained under a strict computational evaluation budget.

## 2. Research Question
"Can context-aware cross-dataset feature experience transfer guide automated feature engineering toward useful feature transformations while reducing the number of expensive candidate evaluations required to reach near-exhaustive predictive performance?"

## 3. Hypotheses H1–H6
- **H1**: Context-aware experience can improve candidate prioritization compared with uninformed/random candidate selection under the same evaluation budget.
- **H2**: Context-aware retrieval can identify more useful transferred transformations than global historical-frequency retrieval.
- **H3**: CAFET can approach the predictive performance of exhaustive AutoFE with substantially fewer expensive candidate evaluations.
- **H4**: Experience accumulated from previously processed datasets becomes useful for subsequently unseen datasets.
- **H5**: Dataset/context similarity provides useful information for predicting feature-transfer usefulness.
- **H6**: Risk-aware transfer can reduce harmful or negative transfer.
