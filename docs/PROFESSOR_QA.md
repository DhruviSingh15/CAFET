# PROFESSOR Q&A

## Q1. What problem does CAFET solve?
Automated feature engineering systems can generate an enormous number of candidate feature transformations, making exhaustive evaluation computationally expensive and often intractable. CAFET (Continual Automated Feature Engineering via Transfer) attempts to solve this problem by organizing feature exploration under strict evaluation budgets. It investigates whether intelligent dataset profiling and historical experience can be used to navigate the vast feature space more efficiently than random search.

## Q2. What is automated feature engineering?
Automated feature engineering is the process of using algorithms to systematically create, evaluate, and select new mathematical representations (features) from raw data. Instead of relying on a human data scientist to manually brainstorm and program transformations (like taking the log of a column or multiplying two columns together), an automated system algorithmically generates and tests these combinations to improve machine learning model performance.

## Q3. What is the role of the experience repository?
The experience repository is a persistent database that stores past dataset profiles, candidate transformations, and their evaluated performance (utility). It acts as the system's memory, tracking historical feature utility. Currently, this data is used to investigate whether past success on similar datasets can inform and optimize future feature searches on new datasets. 

## Q4. Where is the continual learning?
CAFET has the infrastructure for experience accumulation and update, and continual-learning behavior was investigated experimentally. However, the current experiments do not establish a reliable continual-learning performance advantage.

## Q5. What is actually novel?
The novelty lies in the systematic separation of system integration and empirical investigation. The system integration securely weaves dataset intelligence, an experience repository, and budget-aware AutoFE into a unified pipeline. Scientifically, it provides a rigorous empirical investigation into cross-dataset utility prediction and experience-aware representations. We do not claim guaranteed novelty or patentability for the architectural design itself.

## Q6. Why did the utility predictor not work?
Under the tested representations (structural context and micro-context) and dataset pool, the learned signal did not generalize reliably to unseen datasets. Attempting to predict candidate utility using purely structural representation vectors without observing the interaction with the target variable out-of-sample proved insufficient to consistently beat random selection baselines.

## Q7. Why did you continue after the first model failed?
The project systematically tested similarity-weighted history, larger dataset pools, micro-context representations, adaptive search acquisition, independent evaluation, and cross-validation. This demonstrates rigorous hypothesis testing—methodically eliminating experimental confounders like dataset size, feature granularity, and validation variance—rather than simply abandoning the first approach.

## Q8. Why compare against random search?
Because random search provides a simple, budget-matched baseline for determining whether learned acquisition provides useful search efficiency. If an intelligent adaptive system cannot consistently beat random search within the same evaluation budget, the learning mechanism is not providing actionable signal.

## Q9. Did adaptive search beat random search?
Not reliably under the final independent-evaluation protocol. While early inner-validation results appeared promising, strict independent evaluation (Phase 7.6) and cross-validated independent evaluation (Phase 7.7) revealed significant search-to-independent generalization gaps. Random search actually outperformed adaptive search in strict independent out-of-sample metrics.

## Q10. What is the strongest result of the project?
The project produced a complete, reproducible, budget-constrained AutoFE framework and systematically evaluated whether cross-dataset experience could improve candidate selection. The rigorous experimental infrastructure cleanly separated and tested the limitations of unsupervised cross-dataset utility prediction.

## Q11. What would you do next?
Future work should explore target-aware representations, larger dataset pools, stronger meta-learning formulations, and more robust continual-learning evaluation frameworks.

## Q12. Is this patentable?
The current work identifies a system architecture and research direction, but patentability cannot be concluded from these experiments. A prior-art and patent search would be required.
