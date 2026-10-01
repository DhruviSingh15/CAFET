# DEMO SCRIPT

## 0:00–0:45 — Problem

"Automated Feature Engineering is a powerful technique for improving machine learning models by algorithmically generating new features. However, evaluating every possible feature combination is incredibly expensive. In many cases, generating the candidate features is cheap, but repeatedly training a model to evaluate each one quickly becomes intractable. Data scientists need a way to search this massive candidate space efficiently under a strict computational budget."

## 0:45–1:30 — CAFET Idea

"CAFET—Continual Automated Feature Engineering via Transfer—was built to investigate a solution. The core idea is to structure the AutoFE process into a unified pipeline:
We take a Dataset, build a numerical Profile of it, Generate a universe of Candidate features, and then use a strict Search Budget to limit evaluations. We Evaluate the Candidates, Select the Best one, measure its true Test utility, and finally Learn from the results by saving them to an Experience Repository."

## 1:30–3:30 — Live Dashboard

"Let me show you the interactive dashboard that orchestrates this framework."
*(Open dashboard)*
"First, we **select a dataset**—for instance, the diabetes regression task. 
The system instantly computes a **dataset profile**, representing the dataset mathematically. 
Next, we generate the **candidate universe**. These are all the mathematical transformations the system proposes.
We enforce a strict **search budget** to bound our computational cost.
Now, we **run CAFET**. The system orchestrates candidate evaluation within the budget.
Here are the **evaluated candidates** and their inner validation scores.
The system automatically chooses the **selected feature** with the highest search utility.
We display the **search/validation result**, which guides the search process.
Crucially, we separately compute and display the **final held-out test result**, completely preventing target leakage.
Finally, the system stores this result in the **historical experience** repository, shown at the bottom, strictly separate from the current run."

## 3:30–4:30 — Research Architecture

"Under the hood, CAFET isolates its production components from its research algorithms. The core pipeline relies on Dataset Intelligence to encode datasets, and Candidate Representation to encode transformation types. It persists metadata to the Experience Repository. Our central research question was whether we could use Dataset Similarity and historical experience to prioritize Candidate Evaluation intelligently via Budget Allocation, predicting which features would perform best before actually evaluating them."

## 4:30–5:30 — Research Experiments

"To answer this, we systematically tested multiple approaches: from dataset similarity weighting to micro-context feature representations, across 10 diverse datasets. We compared our Adaptive candidate search against a random acquisition baseline.
What we found honestly and rigorously was that under a strict independent-evaluation protocol, cross-dataset structural representations did not establish reliable adaptive improvement. Predicting feature utility strictly from dataset and feature statistics without seeing the target interaction didn't out-perform random baseline evaluations."

## 5:30–6:30 — Contribution

"Despite this negative research finding on out-of-sample utility prediction, CAFET successfully contributes a fully working, modular AutoFE framework. We established strict budget-aware search, a persistent experience infrastructure, and an uncompromised reproducible evaluation protocol. We have clearly identified the limitations of purely unsupervised structural context embeddings in feature evaluation, precisely identifying target-aware meta-learning as the immediate next step for the research community."
