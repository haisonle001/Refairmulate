import torch
import json
import csv
import os
import random
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for query generation"""
    # Model configuration
    model_name: str = 'castorini/doc2query-t5-base-msmarco'
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # Generation parameters
    num_variations: int = 10
    max_length: int = 64
    min_length: int = 20
    do_sample: bool = True
    top_k: int = 20
    top_p: float = 0.9
    temperature: float = 1.0
    num_beams: int = 1
    
    # Generation strategy
    randomize_params: bool = True
    top_k_range: tuple = (5, 20)
    temperature_range: tuple = (0.7, 1.5)
    max_length_range: tuple = (20, 64)
    
    # Quality control
    max_attempts: int = 1000
    max_duplicates_per_variation: int = 3
    min_unique_ratio: float = 0.3  # Minimum ratio of unique variations
    
    # Output configuration
    save_mode: str = 'file'  # 'file', 'folder', 'both'
    output_format: str = 'tsv'  # 'tsv', 'json', 'jsonl'
    batch_size: int = 50  # Number of sequences to generate per batch

class QueryGenerator(ABC):
    """Abstract base class for query generators"""
    
    @abstractmethod
    def generate(self, context: str, **kwargs) -> list[str]:
        """Generate queries from context"""
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the generation model"""
        pass

class T5QueryGenerator(QueryGenerator):
    """T5-based query generator (doc2query, etc.)"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)
    
    def load_model(self):
        """Load T5 model and tokenizer"""
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            logger.info(f"Loading model: {self.config.model_name}")
            # hf token
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, context: str, **kwargs) -> list[str]:
        """Generate queries from document context"""
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(
                context, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Get generation parameters
            gen_params = self._get_generation_params(**kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    **gen_params
                )
            
            # Decode outputs
            queries = []
            for output in outputs:
                query = self.tokenizer.decode(output, skip_special_tokens=True)
                if query.strip():  # Only add non-empty queries
                    queries.append(query.strip())
            
            return queries
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []
    
    def _get_generation_params(self, **kwargs) -> dict:
        """Get generation parameters, with optional randomization"""
        params = {
            'max_length': kwargs.get('max_length', self.config.max_length),
            'min_length': kwargs.get('min_length', self.config.min_length),
            'do_sample': kwargs.get('do_sample', self.config.do_sample),
            'top_k': kwargs.get('top_k', self.config.top_k),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'num_beams': kwargs.get('num_beams', self.config.num_beams),
            'num_return_sequences': kwargs.get('batch_size', self.config.batch_size),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Randomize parameters if enabled
        if self.config.randomize_params and not kwargs.get('disable_randomization', False):
            params.update(self._randomize_params())
        
        return params
    
    def _randomize_params(self) -> dict:
        """Randomize generation parameters for diversity"""
        return {
            'max_length': random.randint(*self.config.max_length_range),
            'top_k': random.randint(*self.config.top_k_range),
            'temperature': random.uniform(*self.config.temperature_range),
        }

class QueryGenerationPipeline:
    """Main pipeline for query generation"""
    
    def __init__(self, generator: QueryGenerator, config: GenerationConfig):
        self.generator = generator
        self.config = config
        
    def generate_variations_for_query(self, 
                                    query_id: str, 
                                    contexts: list[dict], 
                                    num_variations: int = None) -> list[str]:
        """
        Generate query variations for a single query ID
        
        Args:
            query_id: Unique identifier for the query
            contexts: list of context documents/passages
            num_variations: Number of variations to generate (uses config if None)
        
        Returns:
            list of generated query variations
        """
        if num_variations is None:
            num_variations = self.config.num_variations
            
        if not contexts:
            logger.warning(f"No contexts provided for query_id {query_id}")
            return []
        
        unique_variations = set()
        all_variations = []
        attempts = 0
        
        pbar = tqdm(
            total=num_variations, 
            desc=f"Generating variations for {query_id}",
            leave=False
        )
        
        context_idx = 0
        while len(unique_variations) < num_variations and attempts < self.config.max_attempts:
            # Cycle through contexts
            context_data = contexts[context_idx % len(contexts)]
            context_text = self._extract_context_text(context_data)
            context_idx += 1
            
            # Generate batch of queries
            batch_size = min(
                self.config.batch_size,
                num_variations - len(unique_variations)
            )
            
            generated_queries = self.generator.generate(
                context_text,
                batch_size=batch_size
            )
            
            # Process generated queries
            for query in generated_queries:
                attempts += 1
                all_variations.append(query)
                
                if query not in unique_variations:
                    unique_variations.add(query)
                    pbar.update(1)
                
                if len(unique_variations) >= num_variations:
                    break
            
            # Progress check
            if attempts % 100 == 0:
                unique_ratio = len(unique_variations) / max(attempts, 1)
                if unique_ratio < self.config.min_unique_ratio:
                    logger.warning(
                        f"Low uniqueness ratio: {unique_ratio:.3f} "
                        f"({len(unique_variations)}/{attempts})"
                    )
        
        pbar.close()
        
        # Finalize variations list
        final_variations = self._finalize_variations(
            list(unique_variations), all_variations, num_variations
        )
        
        logger.info(
            f"Generated {len(final_variations)} variations for {query_id} "
            f"({len(unique_variations)} unique, {attempts} attempts)"
        )
        
        return final_variations
    
    def _extract_context_text(self, context_data: dict) -> str:
        """Extract text from context data structure"""
        # Handle different context data formats
        if isinstance(context_data, str):
            return context_data
        elif isinstance(context_data, dict):
            # Try common field names
            for field in ['doc_text', 'text', 'content', 'passage', 'document']:
                if field in context_data:
                    return context_data[field]
            # If no standard field, convert to string
            return str(context_data)
        else:
            return str(context_data)
    
    def _finalize_variations(self, 
                           unique_variations: list[str], 
                           all_variations: list[str], 
                           target_count: int) -> list[str]:
        """Finalize the list of variations to meet target count"""
        final_variations = unique_variations.copy()
        
        if len(unique_variations) < target_count:
            logger.warning(
                f"Only generated {len(unique_variations)} unique variations, "
                f"target was {target_count}"
            )
            
            # Add controlled duplicates to reach target
            random.shuffle(all_variations)
            variation_counts = {}
            
            for variation in all_variations:
                if len(final_variations) >= target_count:
                    break
                
                count = variation_counts.get(variation, 0)
                if count < self.config.max_duplicates_per_variation:
                    final_variations.append(variation)
                    variation_counts[variation] = count + 1
        
        return final_variations[:target_count]
    
    def process_dataset(self, 
                       data: dict[str, list[dict]], 
                       output_dir: str,
                       query_ids: list[str] = None,
                       max_queries: int = None) -> dict[str, int]:
        """
        Process entire dataset of query-context pairs
        
        Args:
            data: dictionary mapping query_ids to context lists
            output_dir: Directory to save results
            query_ids: Specific query IDs to process (None = all)
            max_queries: Maximum number of queries to process
        
        Returns:
            dictionary with processing statistics
        """
        # Determine query IDs to process
        if query_ids is None:
            query_ids = list(data.keys())
        
        if max_queries:
            query_ids = query_ids[:max_queries]
        
        # Setup output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process queries
        stats = {'total_queries': 0, 'total_variations': 0, 'failed_queries': 0}
        
        for query_id in tqdm(query_ids, desc="Processing queries"):
            try:
                if query_id not in data:
                    logger.warning(f"Query ID {query_id} not found in data")
                    stats['failed_queries'] += 1
                    continue
                
                variations = self.generate_variations_for_query(
                    query_id, data[query_id]
                )
                
                if variations:
                    self._save_variations(query_id, variations, output_dir)
                    stats['total_variations'] += len(variations)
                else:
                    stats['failed_queries'] += 1
                
                stats['total_queries'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process query {query_id}: {e}")
                stats['failed_queries'] += 1
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    def _save_variations(self, query_id: str, variations: list[str], output_dir: str):
        """Save variations according to configuration"""
        output_path = Path(output_dir)
        
        if self.config.save_mode in ['file', 'both']:
            self._save_to_file(query_id, variations, output_path)
        
        if self.config.save_mode in ['folder', 'both']:
            self._save_to_folder(query_id, variations, output_path)
    
    def _save_to_file(self, query_id: str, variations: list[str], output_path: Path):
        """Save all variations to a single file"""
        if self.config.output_format == 'tsv':
            filename = 'generated_queries.tsv'
            filepath = output_path / filename
            
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                for variation in variations:
                    writer.writerow([query_id, variation])
        
        elif self.config.output_format == 'jsonl':
            filename = 'generated_queries.jsonl'
            filepath = output_path / filename
            
            with open(filepath, 'a', encoding='utf-8') as f:
                for i, variation in enumerate(variations):
                    entry = {
                        'query_id': query_id,
                        'variation_id': i + 1,
                        'query_text': variation
                    }
                    f.write(json.dumps(entry) + '\n')
        
        elif self.config.output_format == 'json':
            filename = 'generated_queries.json'
            filepath = output_path / filename
            
            # Load existing data or create new
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            
            data[query_id] = variations
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_to_folder(self, query_id: str, variations: list[str], output_path: Path):
        """Save each variation to its own file"""
        query_folder = output_path / query_id
        query_folder.mkdir(exist_ok=True)
        
        for i, variation in enumerate(variations, 1):
            if self.config.output_format == 'tsv':
                filename = f'variation_{i}.tsv'
                filepath = query_folder / filename
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([query_id, variation])
            
            elif self.config.output_format == 'json':
                filename = f'variation_{i}.json'
                filepath = query_folder / filename
                
                data = {
                    'query_id': query_id,
                    'variation_id': i,
                    'query_text': variation
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

def load_dataset(file_path: str) -> dict[str, list[dict]]:
    """Load dataset from various formats"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_path.suffix == '.jsonl':
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                query_id = entry['query_id']
                if query_id not in data:
                    data[query_id] = []
                data[query_id].append(entry)
        return data
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def create_generator(generator_type: str, config: GenerationConfig) -> QueryGenerator:
    """Factory function to create generators"""
    if generator_type.lower() in ['t5', 'doc2query']:
        return T5QueryGenerator(config)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

def main():
    parser = argparse.ArgumentParser(description='General Query Generation Pipeline')
    
    # Data arguments
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--query_ids', nargs='*', help='Specific query IDs to process')
    parser.add_argument('--max_queries', type=int, help='Maximum number of queries to process')
    
    # Model arguments
    parser.add_argument('--generator_type', default='t5', choices=['t5', 'doc2query'])
    parser.add_argument('--model_name', default='castorini/doc2query-t5-base-msmarco')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    
    # Generation arguments
    parser.add_argument('--num_variations', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--randomize_params', action='store_true')
    
    # Output arguments
    parser.add_argument('--save_mode', default='file', choices=['file', 'folder', 'both'])
    parser.add_argument('--output_format', default='tsv', choices=['tsv', 'json', 'jsonl'])
    
    # Quality control
    parser.add_argument('--max_attempts', type=int, default=1000)
    parser.add_argument('--min_unique_ratio', type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Create configuration
    config = GenerationConfig(
        model_name=args.model_name,
        device=args.device,
        num_variations=args.num_variations,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        batch_size=args.batch_size,
        randomize_params=args.randomize_params,
        save_mode=args.save_mode,
        output_format=args.output_format,
        max_attempts=args.max_attempts,
        min_unique_ratio=args.min_unique_ratio
    )
    
    # Load data
    logger.info(f"Loading dataset from {args.dataset}")
    data = load_dataset(args.dataset)
    logger.info(f"Loaded {len(data)} queries")
    
    # Create generator and pipeline
    generator = create_generator(args.generator_type, config)
    pipeline = QueryGenerationPipeline(generator, config)
    
    # Process dataset
    logger.info("Starting query generation...")
    stats = pipeline.process_dataset(
        data=data,
        output_dir=args.output_dir,
        query_ids=args.query_ids,
        max_queries=args.max_queries
    )
    
    logger.info("Generation complete!")
    logger.info(f"Final statistics: {stats}")

if __name__ == "__main__":
    main()