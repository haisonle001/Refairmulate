# Re<span style="color:red">fair</span>mulate: A Large-Scale Dataset for Gender-Fair Query Reformulations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXX-b31b1b.svg)]()

**Refairmulate** is the first large-scale dataset specifically designed for fairness-aware query reformulation in Information Retrieval (IR) systems. The dataset contains over **300,000 query pairs** that enable dual-objective optimization of both retrieval effectiveness and gender bias mitigation.
## 🌟 Key Features

- 🎯 **Multi-objective optimization** for both effectiveness and fairness
- 📊 **300,000+ query pairs** derived from MS MARCO passage retrieval corpus
- 🔍 **Three specialized subsets**: Optimal, Effective, and Fair
- ⚖️ **Comprehensive bias metrics**: ARaB variants and LIWC
- 🚀 **Cross-model validation** across SPLADE, SBERT, TCT-ColBERT, and ANCE
- 📈 **Demonstrated improvements**: Up to 76.0% MRR@10 gains and 48.5% bias reduction

## 📋 Table of Contents
 - **Dataset Overview**
 - **Dataset Structure**
 - **Methodology**
 - **Installation**
 - **Usage Examples**
 - **Benchmarking Results**
 - **Citation**
 - **License**

## 📊 Dataset Overview

The Refairmulate dataset addresses a critical gap in fairness-aware query reformulation by providing systematically constructed query pairs that optimize for both retrieval effectiveness and gender bias reduction.

### Dataset Statistics

| Subset        | Query Pairs | Objective                                                  |
| ------------- | ----------- | ---------------------------------------------------------- |
| **Optimal**   | 112,261     | Perfect dual optimization (RR@10 = 1, bias = 0 under BM25) |
| **Effective** | 209,343     | Improved effectiveness and reduced bias                    |
| **Fair**      | 321,604     | Reduced bias with flexible effectiveness constraints       |

Each example contains:

* Original query `q`
* Reformulated query `q'`
* Retrieval effectiveness scores
* Bias scores (ARaB-tc, ARaB-tf, ARaB-bool, LIWC)
* Query group metadata
* Label category (0–4 as defined in paper)

### Key Metrics

- **Effectiveness Metrics**: Mean Reciprocal Rank (MRR@10), Average Precision (AP@10)
- **Bias Metrics**: ARaB-tc, ARaB-tf, ARaB-bool, LIWC
- **Multi-objective Scoring**: S(q, q') = w_e × Δeff(q, q') + w_b × Δbias(q, q')

---
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

---

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

---

## 💻 Installation

### Prerequisites
- Dependencies listed in `requirements.txt`

---

## 📋 Usage Examples

Usage examples for the dataset and codebase are coming soon! In general, you can:
- Download the dataset from the `datasets/` directory.
- Use the provided scripts in `src/` to run query classification, generation, and selection pipelines.
- Evaluate your models using the benchmarking scripts.

---

## 📈 Benchmarking Results

### The overview of our proposed Refairmulate datasets

### Dataset Statistics
| Subset | Query Pairs | MRR@10 Improvement | ARaB-TC Reduction | ARaB-TF Reduction | ARaB-BOOL Reduction | LIWC Reduction | Use Case |
|--------|-------------|-------------------|------------------|------------------|---------------------|----------------|----------|
| **Optimal** | 112,261 | +521.1% | -100.0% | -100.0% | -100.0% | -100.0% | Theoretical upper bound |
| **Effective** | 209,343 | +424.9% | -79.1% | -78.9% | -77.7% | -70.3% | Performance-focused improvements |
| **Fair** | 321,604 | +474.3% | -83.2% | -84.5% | -78.8% | -76.3% | Comprehensive fairness training |


### Cross-Model Performance on Optimal Dataset

The table below shows the performance improvements achieved across different dense retrieval models on a held-out sample of 10,000 queries from the Optimal dataset:

| Model | MRR@10 Improvement | Max ARaB Reduction |
|-------|-------------------|-------------------|
| **SPLADE** | **+76.0%** | **-48.5%** |
| SBERT | +44.7% | -40.8% |
| TCT-ColBERT | +55.4% | -35.6% |
| ANCE | +66.9% | -22.8% |


### Generalization on Human-Annotated Neutral Queries

We evaluate models trained on Refairmulate on two independent human-annotated neutral query sets:

| Query Set | Queries | MRR@10 Improvement | ARaB-TC Reduction | ARaB-TF Reduction | ARaB-BOOL Reduction | LIWC Reduction | Positive Rate | Negative Rate | Neutral Rate |
|-----------|---------|-------------------|------------------|------------------|---------------------|----------------|---------------|---------------|--------------|
| **Small Set** | 215 | +154.6% | -81.5% | -78.6% | -75.6% | -72.1% | 32.2% | 6.7% | 61.1% |
| **Large Set** | 1,765 | +88.7% | -63.3% | -60.0% | -55.6% | -56.3% | 33.3% | 14.6% | 52.1% |

**Interpretation:**
- **Positive Rate**: Percentage of queries showing improvements in both effectiveness and bias
- **Negative Rate**: Percentage of queries showing degradation
- **Neutral Rate**: Percentage of queries with mixed or no significant change

---

## 📖 Citation

If you use Refairmulate in your research, please cite our paper:

```bibtex
@article{TBD,
  title={Refairmulate: A Benchmark Dataset for Gender-Fair Query Reformulations},
  author={TBD},
  journal={TBD},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


