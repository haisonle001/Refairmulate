## **Key Features:**

### **1. Modular Architecture**
- **Abstract `QueryGenerator`**: Base class for different generation models
- **`T5QueryGenerator`**: Specific implementation for T5/doc2query models
- **`QueryGenerationPipeline`**: Main orchestration class
- **`GenerationConfig`**: Centralized configuration management

### **2. Flexible Generator Support**
- Easy to add new model types (GPT, BART, etc.)
- Configurable model parameters
- Automatic device detection and management

### **3. Advanced Generation Features**
- **Parameter randomization** for increased diversity
- **Batch processing** for efficiency
- **Quality control** with uniqueness monitoring
- **Progress tracking** with detailed statistics

### **4. Multiple Output Formats**
- **TSV**: Tab-separated values (original format)
- **JSON**: Structured JSON format
- **JSONL**: JSON Lines for streaming

### **5. Flexible Save Modes**
- **File mode**: All variations in single files
- **Folder mode**: Each variation in separate files
- **Both modes**: Save in both formats

## **Usage Examples:**

### **Basic Usage (equivalent to your original code):**
```bash
python query_generator.py \
    --dataset /path/to/queries.train.gold.doct5.mapping.json \
    --output_dir /path/to/output \
    --num_variations 10 \
    --save_mode file \
    --output_format tsv
```

### **Advanced Usage with Randomization:**
```bash
python query_generator.py \
    --dataset /path/to/data.json \
    --output_dir /path/to/output \
    --num_variations 20 \
    --randomize_params \
    --batch_size 100 \
    --max_attempts 2000 \
    --save_mode both \
    --output_format jsonl
```

### **Custom Model:**
```bash
python query_generator.py \
    --dataset /path/to/data.json \
    --output_dir /path/to/output \
    --model_name "t5-base" \
    --generator_type t5 \
    --temperature 1.2 \
    --top_k 15 \
    --max_length 50
```

### **Process Specific Queries:**
```bash
python query_generator.py \
    --dataset /path/to/data.json \
    --output_dir /path/to/output \
    --query_ids "query1" "query2" "query3" \
    --num_variations 5
```

## **Usage:**

```python
# Create configuration
config = GenerationConfig(
    model_name='castorini/doc2query-t5-base-msmarco',
    num_variations=10,
    randomize_params=True,
    save_mode='file',
    output_format='json'
)

# Create generator and pipeline
generator = T5QueryGenerator(config)
pipeline = QueryGenerationPipeline(generator, config)

# Process data
data = load_dataset('data.json')
stats = pipeline.process_dataset(data, 'output_dir/')
```

## **Easy Extensions:**
### **Add New Generator:**
```python
class GPTQueryGenerator(QueryGenerator):
    def generate(self, context: str, **kwargs) -> List[str]:
        # Implement GPT-based generation
        pass
```

### **Add New Output Format:**
```python
def _save_to_xml(self, query_id: str, variations: List[str], output_path: Path):
    # Implement XML output format
    pass
```