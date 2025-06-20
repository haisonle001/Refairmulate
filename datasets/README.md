# Refairmulate Datasets

This directory contains all the datasets, queries, and evaluation files required to reproduce the experiments in the Refairmulate paper.

##  Directory Structure

```
datasets/
├── data/
│   ├── Optimal.json
│   ├── Effective.json
│   └── Fair.json
├── eval/
│   ├── 215_variations.json
│   ├── 215_bm25.json
│   ├── 1765_variations.json
│   ├── 1765_bm25.json
│   └── dense/
└── queries/
    ├── queries.train.neutral.tsv
    └── queries.train.neutral.mapping.json
```

---

## Subdirectory Descriptions

### `data/`

This directory contains the three main subsets of the Refairmulate dataset in JSON format. Each file consists of query pairs (`org_query_text`, `query_text`) designed for multi-objective optimization.

- **`Optimal.json`**: Contains 112,261 query pairs optimized for perfect effectiveness (MRR@10=1) and fairness (bias=0). This subset represents a theoretical upper bound.
- **`Effective.json`**: Contains 209,343 query pairs focused on maximizing effectiveness gains under fairness constraints.
- **`Fair.json`**: Contains 321,604 query pairs that prioritize significant bias reduction while maintaining or improving effectiveness.

#### Data Format (`.json`)

Each JSON file is a nested dictionary structure where the top level groups queries, and each query contains the original text, reformulated text, and detailed metrics.

```json
{
    "group_id": {
        "query_id": {
            "org_query_text": "text of the original query",
            "query_text": "text of the fair and effective reformulated query",
            "metrics": {
                "Orig_RR10": 1.0,
                "Orig_AP10": 1.0,
                "Orig_LIWC10": 0.0,
                "Orig_ARAB-tc10": 0.0,
                "Orig_ARAB-tf10": 0.0,
                "Orig_ARAB-bool10": 0.0,
                "New_RR10": 1.0,
                "New_AP10": 1.0,
                "New_LIWC10": 0.0,
                "New_ARAB-tc10": 0.0,
                "New_ARAB-tf10": 0.0,
                "New_ARAB-bool20": 0.0,
                "Label": 3,
                "Group": 1
            },
            "score": 1.0
        }
    }
}
```

**Metrics Explanation:**
- **`Orig_*`**: Metrics for the original query (effectiveness: RR10, AP10; bias: LIWC, ARAB variants)
- **`New_*`**: Metrics for the reformulated query  
- **`Label`**: Query classification label
- **`Group`**: Group identifier
- **`score`**: Overall optimization score

### `eval/`

This directory contains files used for the generalization and cross-model performance evaluations.

- **`dense/`**: This subdirectory contains evaluation results for dense retrieval models (ANCE, TCT-ColBERT, SBERT, etc.).
- **`215_*.json` / `1765_*.json`**: These files contain the results for the 215 and 1,765 query generalization sets mentioned in the paper, for both the original BM25 runs and the reformulated variations.

### `queries/`

This directory contains the source queries used to build the Refairmulate dataset.

- **`queries.train.neutral.tsv`**: A tab-separated file containing the gender-neutral queries filtered from the MS MARCO training set. This is the starting point for the generation and selection pipeline.
  - **Format**: `query_id\tquery_text`
- **`queries.train.neutral.mapping.json`**: A JSON file that maps query IDs to their relevant passage IDs from the MS MARCO corpus. This is used to fetch documents for effectiveness and bias calculations.
  - **Format**: `{"query_id": {"query": query_text, "doc_id": doc_id, "doc_text": doc_text}}`