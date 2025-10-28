# Re<span style="color:red">fair</span>mulate: A Benchmark Dataset for Gender-Fair Query Reformulations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXX-b31b1b.svg)]()

**Refairmulate** is the first large-scale benchmark dataset specifically designed for fairness-aware query reformulation in Information Retrieval (IR) systems. The dataset contains over **300,000 query pairs** that enable dual-objective optimization of both retrieval effectiveness and gender bias mitigation.

## 🌟 Key Features

- 🎯 **Multi-objective optimization** for both effectiveness and fairness
- 📊 **300,000+ query pairs** derived from MS MARCO passage retrieval corpus
- 🔍 **Three specialized subsets**: Optimal, Effective, and Fair
- ⚖️ **Comprehensive bias metrics**: ARaB variants and LIWC
- 🚀 **Cross-model validation** across SPLADE, SBERT, TCT-ColBERT, and ANCE
- 📈 **Demonstrated improvements**: Up to 76.0% MRR@10 gains and 94.5% bias reduction

## 📋 Table of Contents

- [Dataset Overview](#-dataset-overview)
- [Dataset Structure](#-dataset-structure)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Benchmarking Results](#-benchmarking-results)
- [Citation](#-citation)
- [License](#-license)

## 📊 Dataset Overview

The Refairmulate dataset addresses a critical gap in fairness-aware query reformulation by providing systematically constructed query pairs that optimize for both retrieval effectiveness and gender bias reduction.

### Dataset Statistics

| Subset | Query Pairs | MRR@10 Improvement | Bias Reduction | Use Case |
|--------|-------------|-------------------|----------------|----------|
| **Optimal** | 112,261 | +855.7% | -100.0% | Theoretical upper bound |
| **Effective** | 209,343 | +426.9% | -388.5% | Performance-focused improvements |
| **Fair** | 321,604 | +476.2% | -506.2% | Comprehensive fairness training |

### Key Metrics

- **Effectiveness Metrics**: Mean Reciprocal Rank (MRR@10), Average Precision (AP@10)
- **Bias Metrics**: ARaB-tc, ARaB-tf, ARaB-bool, LIWC
- **Multi-objective Scoring**: S(q, q') = w_e × Δeff(q, q') + w_b × Δbias(q, q')

## 🏗️ Dataset Structure

The Refairmulate dataset is partitioned into three subsets, each targeting distinct fairness-effectiveness trade-offs:

### 📈 Optimal Subset
- **Size**: 112,261 query pairs
- **Objective**: Perfect dual optimization (MRR@10 = 1, bias = 0)
- **Use case**: Theoretical upper bound for fairness-performance trade-offs

### ⚡ Effective Subset  
- **Size**: 209,343 query pairs
- **Objective**: Maximal performance improvement under fairness constraints
- **Use case**: Targeted improvements for specific query categories

### ⚖️ Fair Subset
- **Size**: 321,604 query pairs
- **Objective**: Measurable bias reduction with flexible performance requirements
- **Use case**: Comprehensive fairness-aware training and evaluation

### Query Group Classification

We categorize queries into 4 groups based on their effectiveness (eff(q, D_q)) and bias (bias(q, D_q)) relative to thresholds (θ_eff and θ_bias):

- **Group 1**: High Effectiveness, Low Bias - Minimal reformulation needed
- **Group 2**: High Effectiveness, High Bias - Focus on bias reduction while preserving effectiveness
- **Group 3**: Low Effectiveness, Low Bias - Focus on effectiveness improvement
- **Group 4**: Low Effectiveness, High Bias - Comprehensive reformulation for both issues

## 🔬 Methodology

### Algorithm

The following pseudocode outlines the Refairmulate process for fair and effective query reformulation:

```plaintext
Algorithm: Refairmulate - Fair and Effective Query Reformulation

Input: Query set Q, relevant documents D_q for each q in Q
Output: Reformulated query pairs QP = {(q, q') | q in Q, q' in Q'}

1. Initialize QP as an empty set
2. For each query q in Q:
    a. If C(q) ≠ 0, skip q  // Skip biased queries
    b. Compute bias(q, D_q) and eff(q, D_q), then categorize q
    c. Generate variants V_q = G(q, D_q)
    d. For each variant v_q^(i) in V_q:
        i. Compute bias(v_q^(i), D_v_q^(i)) and eff(v_q^(i), D_v_q^(i))
        ii. Calculate score S(q, v_q^(i)) = w_e * Δeff + w_b * Δbias
    e. Select q' = argmax_{v_q^(i) in V_q} S(q, v_q^(i))
    f. Add (q, q') to QP
3. Return QP
```

Our construction pipeline follows a three-stage approach: **Classify → Generate → Select**

### 1. Query Classification
- BERT-based filtering for gender-neutral queries
- Removes queries with inherent gender bias
- Ensures bias measurements reflect system behavior

### 2. Query Generation
- LLM-based reformulation with diverse candidates
- Uses transformer models fine-tuned on query-document pairs
- Generates multiple variations per original query

### 3. Multi-objective Selection
- Optimization balancing effectiveness and fairness
- Group-specific selection criteria
- Comprehensive evaluation using multiple bias metrics

```python
# Multi-objective scoring function
S(q, q') = w_e × Δeff(q, q') + w_b × Δbias(q, q')

where:
- Δeff: Effectiveness improvement
- Δbias: Bias reduction  
- w_e, w_b: Configurable weights
```

## 💻 Installation

### Prerequisites
- Dependencies listed in `requirements.txt`

## 📈 Benchmarking Results

### The overview of our proposed Refairmulate datasets

| Dataset | MRR@10 Source | MRR@10 Destination | MRR@10 Improv. (%) | ARaB-tf@10 Source | ARaB-tf@10 Destination | ARaB-tf@10 Improv. (%) | ARaB-tc@10 Source | ARaB-tc@10 Destination | ARaB-tc@10 Improv. (%) | ARaB-bool@10 Source | ARaB-bool@10 Destination | ARaB-bool@10 Improv. (%) | LIWC@10 Source | LIWC@10 Destination | LIWC@10 Improv. (%) |
|---------|---------------|--------------------|--------------------|-------------------|------------------------|------------------------|-------------------|------------------------|------------------------|---------------------|--------------------------|--------------------------|----------------|---------------------|---------------------|
| **Optimal** | 0.161 | 1.000 | 855.715 | 0.0627 | 0.000 | 100.000 | 0.033 | 0.000 | 100.000 | 0.0348 | 0.000 | 100.000 | 0.132 | 0.000 | 100.000 |
| **Effective** | 0.081 | 0.425 | 426.866 | 0.139 | 0.029 | 388.453 | 0.071 | 0.015 | 370.731 | 0.070 | 0.0156 | 350.393 | 0.273 | 0.081 | 235.330 |
| **Fair** | 0.109 | 0.626 | 476.180 | 0.113 | 0.019 | 506.243 | 0.058 | 0.009 | 488.916 | 0.058 | 0.058 | 469.897 | 0.224 | 0.053 | 322.279 |


### Cross-Model Performance

| Model | MRR@10 Improvement | ARaB Reduction |
|-------|-------------------|----------------|
| SPLADE | +76.0% | -94.5% |
| SBERT | +68.2% | -89.1% |
| TCT-ColBERT | +71.5% | -91.8% |
| ANCE | +65.9% | -87.3% |

### Generalization Performance

| Query Set | Effectiveness | Bias Reduction | Positive Rate |
|-----------|--------------|----------------|---------------|
| 215 queries | +154.6% | -441.3% | 79.07% |
| 1,765 queries | +88.7% | -172.6% | 75.58% |

## 🛠️ Repository Structure

```
refairmulate/
├── datasets/                       # Dataset files and evaluation data
│   ├── data/                       # Main dataset files
│   ├── eval/                       # Evaluation datasets
│   └── queries/                    # Query collections
├── resources/                      # Pre-computed resources and models
│   ├── outputcollection_neutralityscores.tsv
│   ├── liwccollection_bias.pkl
│   ├── msmarco_passage_docs_bias_tf.pkl
│   ├── msmarco_passage_docs_bias_bool.pkl
│   └── msmarco_passage_docs_bias_tc.pkl
├── src/
│   ├── classification/             # Query classification pipeline
│   │   ├── data/                   # Classification data
│   │   ├── model/                  # Classification models
│   │   ├── src/                    # Classification source code
│   │   ├── scripts/                # Classification scripts
│   │   ├── data_loader/            # Data loading utilities
│   │   └── requirements.txt        # Classification dependencies
│   ├── generation/                 # Query generation pipeline
│   │   ├── generate.py             # Main generation script
│   │   └── README.md               # Generation documentation
│   ├── selection/                  # Multi-objective selection
│   │   ├── src/                    # Selection source code
│   │   ├── scripts/                # Selection scripts
│   │   └── README.md               # Selection documentation
│   └── benchmark/                  # Benchmarking and evaluation
│       └── cross-encoder/          # Cross-encoder models
├── .gitattributes                  # Git attributes
└── README.md                       # This file
```

## 🎯 Use Cases

### Research Applications
- **Ablation studies**: Analyze fairness-performance trade-offs
- **Bias measurement**: Benchmark existing systems for gender bias
- **Model comparison**: Evaluate different retrieval architectures

### Practical Applications
- **Search engine optimization**: Improve both relevance and fairness
- **Content recommendation**: Reduce algorithmic bias in recommendations
- **Academic research**: Support reproducible fairness studies
- **Industry applications**: Deploy fairer IR systems


## 📖 Citation

If you use Refairmulate in your research, please cite our paper:

```bibtex
@article{le2025refairmulate,
  title={Refairmulate: A Benchmark Dataset for Gender-Fair Query Reformulations},
  author={Hai Son, Le and Seyedsalehi, Shirin and Kermani, Morteza Zihayat and Bagheri, Ebrahim},
  journal={TBD},
  year={2025}
}
```

## 📧 Contact

- **Hai Son Le** - Toronto Metropolitan University
- **Shirin Seyedsalehi** - Toronto Metropolitan University  
- **Morteza Zihayat Kermani** - Toronto Metropolitan University
- **Ebrahim Bagheri** - University of Toronto

For questions and support, please open an issue on GitHub or contact the authors.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the MS MARCO passage retrieval corpus
- Evaluation conducted using Anserini toolkit
- Special thanks to the fairness-aware IR research community

## 🔗 Related Work

- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Anserini Toolkit](https://github.com/castorini/anserini)
- [Pyserini Toolkit](https://github.com/castorini/pyserini)
- [SPLADE](https://github.com/naver/splade)
- [Sentence-BERT](https://github.com/UKPLab/sentence-transformers)

---
