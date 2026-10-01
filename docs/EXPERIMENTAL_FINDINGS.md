# EXPERIMENTAL FINDINGS

| Phase | Question | Result | Status |
|---|---|---|---|
| Phase 2 | Can we establish a random budget-constrained AutoFE baseline? | Random search provides a robust baseline for budget-limited exploration. | Completed |
| Phase 3 | Can dataset profiles be used for structural dataset intelligence? | Implemented numerical profiles capable of distinguishing raw dataset characteristics. | Implemented |
| Phase 4 | Can feature utility experience be systematically stored? | Created SQLite-based Experience Repository tracking datasets, features, and utility. | Demonstrated |
| Phase 5 | Does naive dataset similarity predict candidate utility? | Target-free dataset similarity failed to reliably predict utility across diverse contexts. | Negative result |
| Phase 6 | Can contextual embeddings represent candidate transformations? | Successfully encoded structural and transformation characteristics into dense vectors. | Implemented |
| Phase 7.1 | Does similarity-weighted experience improve utility prediction? | Similarity modulation did not overcome the structural embedding limitations. | Negative result |
| Phase 7.2 | How does the utility predictor perform across a larger dataset pool? | Expanded to 10 datasets; structural utility prediction failed to generalize reliably out-of-sample. | Investigated |
| Phase 7.3 | Can micro-context embeddings capture finer feature interactions? | Handcrafted interaction statistics improved representation but not out-of-sample prediction. | Completed |
| Phase 7.4 | Does combining macro and micro context improve surrogate models? | Surrogate error remained high out-of-sample. | Negative result |
| Phase 7.5 | Can adaptive acquisition (UCB) outperform random search? | Initial validation showed promise under low budgets, but optimization was highly split-dependent. | Investigated |
| Phase 7.6 | Does the adaptive advantage hold under strict independent evaluation? | Strict evaluation revealed a large generalization gap; Adaptive did not reliably beat Random. | Negative result |
| Phase 7.7 | Does cross-validated utility improve independent generalization? | Cross-validated utility did not produce reliable adaptive-search improvement over random search under independent evaluation. | Negative result |
