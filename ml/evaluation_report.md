# Match-Scoring Model Evaluation

**This report is generated from a synthetic demo dataset (`ml/data/`), not real manually
labeled postings.** Spec section 3.2 calls for manually labeling 150-300 *real* postings
against the candidate's *real* resume — that step requires a human and real scraped
data, and can't be fabricated. `ml/synthetic_data.py` generates a structurally realistic
stand-in so the training/evaluation pipeline itself is provably correct end-to-end. See
`ml/README.md` for how to regenerate this report from a real labeled dataset — the
numbers below should not be quoted as real model performance (e.g. in a resume bullet).

Re-ranker model: **logistic_regression** · Test set size: 60 (held out, unseen during training) · K = 10

| Approach | Precision@10 | Recall@10 | ROC-AUC |
|---|---|---|---|
| Keyword / BM25 | 0.400 | 0.154 | 0.540 |
| Embedding similarity only | 0.700 | 0.269 | 0.689 |
| Embedding + re-ranker | 0.900 | 0.346 | 0.749 |

Embedding + re-ranker vs. embedding-only baseline: **+0.060 ROC-AUC**.
