# Retrieval

## **Key Features:**

### **1. Modular Architecture**
- **Abstract `Retriever`**: Base class for all retrievers
- **`BM25Retriever`**: Sparse retrieval using Anserini/Pyserini
- **`DenseRetriever`**: Dense retrieval supporting multiple models
- **`RetrievalPipeline`**: Main orchestration class

### **2. Supported Retrievers**
- **BM25/Sparse**: Traditional BM25 with configurable parameters
- **ANCE**: Dense retrieval with ANCE encoder
- **TCT-ColBERT**: Late interaction dense retrieval
- **Sentence-BERT**: Sentence transformer-based retrieval
- **SPLADE**: Sparse learned dense retrieval
- **BGE**: BAAI general embedding model

### **3. Flexible Input/Output**
- **Input formats**: TSV, JSON, Dataset directories
- **Output formats**: TREC, JSON, TSV
- **Batch processing** for large query sets
- **Dataset subset processing** (optimal, effective, fair)

## **Usage Examples:**

### **1. Basic TSV Query Retrieval (BM25):**
```bash
python retrieval.py \
    --input /path/to/queries.tsv \
    --output /path/to/run.trec \
    --retriever_type bm25 \
    --hits 1000 \
    --bm25_k1 0.82 \
    --bm25_b 0.68
```

### **2. Dense Retrieval (ANCE):**
```bash
python retrieval.py \
    --input /path/to/queries.tsv \
    --output /path/to/run.trec \
    --retriever_type ance \
    --hits 1000 \
    --batch_size 500
```

### **3. Process Dataset Subsets:**
```bash
python retrieval.py \
    --input /path/to/dataset_dir/ \
    --output /path/to/output_dir/ \
    --retriever_type tct \
    --input_format dataset \
    --subset_names optimal effective fair
```

### **4. Custom Dense Model:**
```bash
python retrieval.py \
    --input /path/to/queries.json \
    --output /path/to/results.json \
    --retriever_type dense \
    --model_name sbert \
    --encoder_name "sentence-transformers/msmarco-distilbert-base-v3" \
    --output_format json
```

### **5. BGE with Custom Parameters:**
```bash
python retrieval.py \
    --input /path/to/queries.tsv \
    --output /path/to/run.trec \
    --retriever_type bge \
    --ef_search 2000 \
    --hits 1000
```

## **Input Format Support:**

### **TSV Input (as requested):**
```tsv
query_id_1    query text 1
query_id_2    query text 2
```

### **JSON Input:**
```json
{
    "query_id_1": "query text 1",
    "query_id_2": "query text 2"
}
```

### **Dataset Input (Refairmulate format):**
```json
{
    "optimal": {
        "query_id_1": {
            "original_query": "original text",
            "reformulated_query": "reformulated text"
        }
    }
}
```

## **Output Formats:**

### **TREC Format:**
```
query_id Q0 doc_id rank score run_tag
```

### **JSON Format:**
```json
{
    "query_id": [
        {"docid": "doc1", "rank": 1, "score": 0.95},
        {"docid": "doc2", "rank": 2, "score": 0.88}
    ]
}
```

## **Integration with Your Pipeline:**

### **1. After Query Generation:**
```bash
# Generate queries
python generate.py --dataset data.json --output_dir generated_queries/

# Retrieve with BM25
python retrieval.py \
    --input generated_queries/queries.tsv \
    --output results/run.bm25.trec \
    --retriever_type bm25

# Retrieve with dense
python retrieval.py \
    --input generated_queries/queries.tsv \
    --output results/run.ance.trec \
    --retriever_type ance
```

### **2. For Dataset Subsets:**
```bash
# Process Refairmulate dataset
python retrieval.py \
    --input dataset/refairmulate/ \
    --output retrieval_results/ \
    --retriever_type bm25 \
    --input_format dataset
```

## **Key Advantages:**

1. **Unified Interface**: Same API for all retrievers
2. **TSV Support**: Direct support for your TSV format
3. **Batch Processing**: Efficient handling of large query sets
4. **Configurable**: All parameters can be tuned
5. **Extensible**: Easy to add new retriever types
6. **Dataset-Aware**: Built-in support for Refairmulate format
7. **Multiple Outputs**: TREC, JSON, TSV formats supported



# Evaluation

# Evaluation Module

## **Key Features:**

### **1. Modular Architecture**
- **`RetrievalMetricsCalculator`**: Computes retrieval effectiveness metrics (RR, AP)
- **`LIWCBiasCalculator`**: Computes LIWC linguistic bias scores
- **`ARaBCalculator`**: Computes Average Ranking-aware Bias (ARaB) scores with tc/tf/bool variants
- **`FaiRRCalculator`**: Computes FaiRR and NFaiRR fairness metrics
- **`EvaluationPipeline`**: Main orchestration class with configurable components

### **2. Flexible Input Support**
- **TREC format**: Standard TREC run files (space-separated format)
- **JSON format**: JSON output from retrieval modules
- **Automatic detection**: Based on file extension (.json vs others)
- **Batch processing**: Multiple run files in single command

### **3. Comprehensive Metrics**
- **Effectiveness**: Reciprocal Rank (RR), Average Precision (AP) at configurable cutoffs
- **Bias Metrics**: 
  - LIWC linguistic bias scores
  - ARaB (tc/tf/bool variants) - ranking-aware bias
  - FaiRR/NFaiRR - fairness-aware ranking metrics
- **Configurable cutoffs**: Default @10, @20, fully customizable

### **4. Configurable Resource Management**
- **Hardcoded default paths** for your specific environment
- **Command-line overrides** for all resource files
- **Graceful error handling** for missing resources
- **Detailed logging** for debugging

### **5. Multiple Output Formats**
- **JSON**: Structured data format for programmatic use
- **TSV**: Tab-separated format for spreadsheet analysis
- **Both formats**: Simultaneous output in both formats
- **Per-query results**: Individual query metrics
- **Aggregated results**: Averaged metrics across all queries

## **Usage Examples:**

### **1. Basic Evaluation (TREC input):**
```bash
python evaluation_module.py \
    --runs /path/to/run.trec \
    --qrels /path/to/qrels.train.tsv \
    --output_dir /path/to/results/ \
    --cutoffs 10 20
```

### **2. Evaluate Multiple Runs:**
```bash
python evaluation_module.py \
    --runs run1.trec run2.trec run3.trec \
    --qrels qrels.train.tsv \
    --output_dir evaluation_results/ \
    --output_format both
```

### **3. JSON Input from Retrieval Module:**
```bash
python evaluation_module.py \
    --runs retrieval_results.json \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --cutoffs 10 20 50
```

### **4. Custom Resource Paths:**
```bash
python evaluation_module.py \
    --runs run.trec \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --liwc_dict_path /custom/path/liwccollection_bias.pkl \
    --docs_bias_tc_path /custom/path/docs_bias_tc.pkl \
    --docs_bias_tf_path /custom/path/docs_bias_tf.pkl \
    --docs_bias_bool_path /custom/path/docs_bias_bool.pkl \
    --collection_neutrality_path /custom/path/neutrality_scores.tsv
```

### **5. Selective Metric Computation:**
```bash
python evaluation_module.py \
    --runs run.trec \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --compute_retrieval \
    --compute_liwc \
    --compute_arab \
    # Note: All metrics are enabled by default
```

### **6. Disable Specific Metrics:**
```bash
python evaluation_module.py \
    --runs run.trec \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --no-compute_fairr  # Would need to modify argparse for this
```

### **7. With Background Run for FaiRR:**
```bash
python evaluation_module.py \
    --runs reformulated_run.trec \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --background_run baseline_run.trec
```

### **8. Custom Cutoffs and Output Options:**
```bash
python evaluation_module.py \
    --runs run.trec \
    --qrels qrels.train.tsv \
    --output_dir results/ \
    --cutoffs 5 10 20 50 100 \
    --output_format tsv \
    --save_per_query \
    --save_aggregated
```

## **Command Line Arguments:**

### **Required Arguments:**
- `--runs`: One or more run files to evaluate (TREC or JSON format)
- `--output_dir`: Directory to save evaluation results

### **Optional Arguments:**
- `--qrels`: Path to qrels file (relevance judgments)
- `--background_run`: Background run file for FaiRR computation
- `--cutoffs`: Cutoff ranks for evaluation (default: 10 20)
- `--output_format`: Output format - json, tsv, or both (default: json)
- `--save_per_query`: Save per-query results (default: True)
- `--save_aggregated`: Save aggregated results (default: True)

### **Metric Control:**
- `--compute_retrieval`: Compute RR/AP metrics (default: True)
- `--compute_liwc`: Compute LIWC bias scores (default: True)
- `--compute_arab`: Compute ARaB bias scores (default: True)
- `--compute_fairr`: Compute FaiRR/NFaiRR scores (default: True)

### **Resource Paths (with your environment defaults):**
- `--liwc_dict_path`: Path to LIWC dictionary pickle file
  - Default: `/mnt/data/son/Refairmulate/resources/liwccollection_bias.pkl`
- `--docs_bias_tc_path`: Path to document bias (tc variant) pickle file
  - Default: `/mnt/data/son/Refairmulate/resources/msmarco_passage_docs_bias_tc.pkl`
- `--docs_bias_tf_path`: Path to document bias (tf variant) pickle file
  - Default: `/mnt/data/son/Refairmulate/resources/msmarco_passage_docs_bias_tf.pkl`
- `--docs_bias_bool_path`: Path to document bias (bool variant) pickle file
  - Default: `/mnt/data/son/Refairmulate/resources/msmarco_passage_docs_bias_bool.pkl`
- `--collection_neutrality_path`: Path to collection neutrality scores
  - Default: `/mnt/data/son/Refairmulate/resources/outputcollection_neutralityscores.tsv`

## **Output Structure:**

### **Per-Query Results (JSON):**
```json
{
    "query_id_1": {
        "RR10": 0.5,
        "AP10": 0.3,
        "RR20": 0.5,
        "AP20": 0.25,
        "LIWC10": 0.0234,
        "LIWC20": 0.0198,
        "ARAB-tc10": 0.0123,
        "ARAB-tf10": 0.0087,
        "ARAB-bool10": 0.0156,
        "ARAB-tc20": 0.0145,
        "ARAB-tf20": 0.0098,
        "ARAB-bool20": 0.0167,
        "FaiRR10": 0.8472,
        "NFaiRR10": 0.9234
    }
}
```

### **Aggregated Results (JSON):**
```json
{
    "avg_RR10": 0.45123,
    "avg_AP10": 0.32456,
    "avg_RR20": 0.43998,
    "avg_AP20": 0.30123,
    "avg_LIWC10": 0.02134,
    "avg_LIWC20": 0.01987,
    "avg_ARAB-tc10": 0.01234,
    "avg_ARAB-tf10": 0.00987,
    "avg_ARAB-bool10": 0.01456
}
```

### **Output Files Generated:**
- `{run_name}_per_query.json/tsv`: Individual query results
- `{run_name}_aggregated.json/tsv`: Averaged metrics
- Console output with summary statistics


# Selection 

## **Step 1: Query Selection (`QuerySelector`)**
- **No scoring function used** - just labels all variants and saves to TSV
- Processes all variants for each query in each group
- Uses `compare_and_label()` to assign labels (0-7)
- Saves everything to TSV files (`group_1_results.tsv`, etc.)

## **Step 2: Dataset Building (`DatasetBuilder`)**
- **Uses scoring function** to select the best variants from TSV files
- Reads the TSV files created in Step 1
- Filters by expected labels for each group:
  - Group 1: label 3 (no change)
  - Group 2: label 1 (improve bias, equal performance)  
  - Group 3: label 2 (improve performance, equal bias)
  - Group 4: label 0 (improve both)
- Applies **scoring function** `S(q,v) = w_e * Δeff + w_b * Δbias`
- Selects best variant per query based on score
- Saves final datasets as JSON files

## **Key Parameters from Paper:**
- `theta_eff`, `theta_bias`: Thresholds for group categorization
- `w_e`, `w_b`: Weights in scoring function (Equation 1)
- `beta_eff`, `beta_bias`: Improvement thresholds for labeling

## **Usage:**
```python
config = RefairmulateConfig(
    w_e=1.5,  # Higher weight for effectiveness
    w_b=0.8,  # Lower weight for bias
    theta_eff=0.9,  # Custom effectiveness threshold
)

# Step 1: Selection (no scoring)
selector = QuerySelector(config)
selector.run_selection()

# Step 2: Build dataset (with scoring)
builder = DatasetBuilder(config)
final_dataset, perfect_dataset = builder.build_dataset()
```
