import os
import csv
import json
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for retrieval systems"""
    # General settings
    hits: int = 1000
    output_format: str = 'trec'  # 'trec', 'json', 'tsv'
    parallelism: int = 1
    batch_size: int = 1000
    
    # BM25 parameters
    bm25_k1: float = 0.82
    bm25_b: float = 0.68
    
    # Dense retrieval parameters
    ef_search: int = 1000
    
    # Model-specific settings
    model_name: str = None
    index_path: str = None
    encoder_name: str = None

class Retriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def search(self, query: str, hits: int = 1000) -> list[dict]:
        """Search for documents given a query"""
        pass
    
    @abstractmethod
    def load_index(self):
        """Load the search index"""
        pass

class BM25Retriever(Retriever):
    """BM25 sparse retriever using Anserini/Pyserini"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.searcher = None
        self.load_index()
    
    def load_index(self):
        """Load BM25 index"""
        try:
            from pyserini.search.lucene import LuceneSearcher
            
            if self.config.index_path:
                self.searcher = LuceneSearcher(self.config.index_path)
            else:
                # Use prebuilt index
                self.searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
            
            # Set BM25 parameters
            self.searcher.set_bm25(self.config.bm25_k1, self.config.bm25_b)
            logger.info(f"BM25 retriever loaded with k1={self.config.bm25_k1}, b={self.config.bm25_b}")
            
        except ImportError:
            logger.error("Pyserini not installed. Please install with: pip install pyserini")
            raise
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            raise
    
    def search(self, query: str, hits: int = 1000) -> list[dict]:
        """Search using BM25"""
        try:
            pyserini_hits = self.searcher.search(query, hits)
            
            results = []
            for i, hit in enumerate(pyserini_hits):
                results.append({
                    'docid': hit.docid,
                    'rank': i + 1,
                    'score': float(hit.score),
                    'content': getattr(hit, 'contents', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed for query '{query}': {e}")
            return []

class DenseRetriever(Retriever):
    """Dense retriever using various models"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.searcher = None
        self.encoder = None
        self.load_index()
    
    def load_index(self):
        """Load dense index and encoder"""
        try:
            model_name = self.config.model_name.lower()
            
            if 'ance' in model_name:
                self._load_ance()
            elif 'tct' in model_name or 'colbert' in model_name:
                self._load_tct_colbert()
            elif 'sbert' in model_name or 'sentence' in model_name:
                self._load_sbert()
            elif 'splade' in model_name:
                self._load_splade()
            elif 'bge' in model_name:
                self._load_bge()
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")
                
            logger.info(f"Dense retriever loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load dense retriever: {e}")
            raise
    
    def _load_ance(self):
        """Load ANCE retriever"""
        from pyserini.encode import AnceQueryEncoder
        from pyserini.search.faiss import FaissSearcher
        
        self.encoder = AnceQueryEncoder('castorini/ance-msmarco-passage')
        if self.config.index_path:
            self.searcher = FaissSearcher(self.config.index_path, self.encoder)
        else:
            self.searcher = FaissSearcher.from_prebuilt_index(
                'msmarco-v1-passage.ance',
                'castorini/ance-msmarco-passage'
            )
    
    def _load_tct_colbert(self):
        """Load TCT-ColBERT retriever"""
        from pyserini.encode import TctColBertQueryEncoder
        from pyserini.search.faiss import FaissSearcher
        
        encoder_name = self.config.encoder_name or 'castorini/tct_colbert-v2-hnp-msmarco'
        self.encoder = TctColBertQueryEncoder(encoder_name)
        
        if self.config.index_path:
            self.searcher = FaissSearcher(self.config.index_path, self.encoder)
        else:
            self.searcher = FaissSearcher.from_prebuilt_index(
                'msmarco-v1-passage.tct_colbert-v2-hnp',
                self.encoder
            )
    
    def _load_sbert(self):
        """Load Sentence-BERT retriever"""
        from pyserini.search.faiss import FaissSearcher
        
        encoder_name = self.config.encoder_name or 'sentence-transformers/msmarco-distilbert-base-v3'
        if self.config.index_path:
            self.searcher = FaissSearcher(self.config.index_path, encoder_name)
        else:
            self.searcher = FaissSearcher.from_prebuilt_index(
                'msmarco-v1-passage.sbert',
                encoder_name
            )
    
    def _load_splade(self):
        """Load SPLADE retriever"""
        from pyserini.search.lucene import LuceneImpactSearcher
        
        encoder_name = self.config.encoder_name or 'naver/splade-cocondenser-ensembledistil'
        if self.config.index_path:
            self.searcher = LuceneImpactSearcher(self.config.index_path, encoder_name)
        else:
            self.searcher = LuceneImpactSearcher.from_prebuilt_index(
                'msmarco-v1-passage.splade-pp-ed',
                encoder_name
            )
    
    def _load_bge(self):
        """Load BGE retriever"""
        from pyserini.search.lucene import LuceneHnswDenseSearcher
        
        if self.config.index_path:
            self.searcher = LuceneHnswDenseSearcher(
                self.config.index_path,
                'BgeBaseEn15',
                ef_search=self.config.ef_search
            )
        else:
            self.searcher = LuceneHnswDenseSearcher.from_prebuilt_index(
                'msmarco-v1-passage.bge-base-en-v1.5.hnsw',
                ef_search=self.config.ef_search,
                encoder='BgeBaseEn15'
            )
    
    def search(self, query: str, hits: int = 1000) -> list[dict]:
        """Search using dense retriever"""
        try:
            pyserini_hits = self.searcher.search(query, hits)
            
            results = []
            for i, hit in enumerate(pyserini_hits):
                results.append({
                    'docid': hit.docid,
                    'rank': i + 1,
                    'score': float(hit.score),
                    'content': getattr(hit, 'contents', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed for query '{query}': {e}")
            return []

class RetrievalPipeline:
    """Main retrieval pipeline"""
    
    def __init__(self, retriever: Retriever, config: RetrievalConfig):
        self.retriever = retriever
        self.config = config
    
    def load_queries_from_tsv(self, file_path: str) -> dict[str, str]:
        """Load queries from TSV file"""
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    query_id = row[0].strip()
                    query_text = row[1].strip()
                    queries[query_id] = query_text
        
        logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries
    
    def load_queries_from_json(self, file_path: str) -> dict[str, str]:
        """Load queries from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Check if it's a direct mapping
            if all(isinstance(v, str) for v in data.values()):
                queries = data
            else:
                # Extract query_text from nested structure
                queries = {}
                for qid, qdata in data.items():
                    if isinstance(qdata, dict):
                        if 'query_text' in qdata:
                            queries[qid] = qdata['query_text']
                        elif 'reformulated_query' in qdata:
                            queries[qid] = qdata['reformulated_query']
                        else:
                            logger.warning(f"No query text found for {qid}")
                    else:
                        queries[qid] = str(qdata)
        else:
            raise ValueError("Unsupported JSON format")
        
        logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries
    
    def retrieve_batch(self, 
                      queries: dict[str, str], 
                      output_path: str,
                      run_tag: str = "retrieval") -> dict[str, list[dict]]:
        """
        Retrieve documents for a batch of queries
        
        Args:
            queries: dictionary mapping query IDs to query texts
            output_path: Path to save results
            run_tag: Tag to identify the run
        
        Returns:
            dictionary mapping query IDs to retrieved documents
        """
        all_results = {}
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Process queries in batches
        query_items = list(queries.items())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.config.output_format == 'trec':
                writer = csv.writer(f, delimiter=' ')
            elif self.config.output_format == 'tsv':
                writer = csv.writer(f, delimiter='\t')
            else:
                writer = None
            
            for i in tqdm(range(0, len(query_items), self.config.batch_size), 
                         desc="Processing query batches"):
                batch = query_items[i:i + self.config.batch_size]
                
                for query_id, query_text in tqdm(batch, desc="Retrieving", leave=False):
                    try:
                        # Retrieve documents
                        results = self.retriever.search(query_text, self.config.hits)
                        all_results[query_id] = results
                        
                        # Write results
                        if self.config.output_format == 'trec':
                            self._write_trec_results(writer, query_id, results, run_tag)
                        elif self.config.output_format == 'tsv':
                            self._write_tsv_results(writer, query_id, results, run_tag)
                        
                    except Exception as e:
                        logger.error(f"Failed to retrieve for query {query_id}: {e}")
                        all_results[query_id] = []
        
        # Save JSON format if requested
        if self.config.output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Retrieval complete. Results saved to {output_path}")
        return all_results
    
    def _write_trec_results(self, 
                           writer: csv.writer, 
                           query_id: str, 
                           results: list[dict], 
                           run_tag: str):
        """Write results in TREC format"""
        for result in results:
            writer.writerow([
                query_id,
                'Q0',
                result['docid'],
                result['rank'],
                result['score'],
                run_tag
            ])
    
    def _write_tsv_results(self, 
                          writer: csv.writer, 
                          query_id: str, 
                          results: list[dict], 
                          run_tag: str):
        """Write results in TSV format"""
        for result in results:
            writer.writerow([
                query_id,
                result['docid'],
                result['rank'],
                result['score'],
                run_tag
            ])
    
    def retrieve_dataset_subsets(self, 
                               dataset_path: str, 
                               output_dir: str,
                               subset_names: list[str] = None) -> dict[str, str]:
        """
        Retrieve for dataset subsets (optimal, effective, fair)
        
        Args:
            dataset_path: Path to dataset JSON file or directory
            output_dir: Output directory for TREC files
            subset_names: list of subset names to process
        
        Returns:
            dictionary mapping subset names to output file paths
        """
        if subset_names is None:
            subset_names = ['optimal', 'effective', 'fair']
        
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        for subset_name in subset_names:
            logger.info(f"Processing {subset_name} subset...")
            
            # Load subset data
            if dataset_path.is_file():
                # Single file with all subsets
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if subset_name in data:
                    subset_data = data[subset_name]
                else:
                    logger.warning(f"Subset {subset_name} not found in {dataset_path}")
                    continue
            else:
                # Directory with separate files
                subset_file = dataset_path / f"{subset_name}.json"
                if not subset_file.exists():
                    logger.warning(f"Subset file not found: {subset_file}")
                    continue
                
                with open(subset_file, 'r', encoding='utf-8') as f:
                    subset_data = json.load(f)
            
            # Extract queries for both original and reformulated
            original_queries = {}
            reformulated_queries = {}
            
            for query_id, query_data in subset_data.items():
                if isinstance(query_data, dict):
                    # Extract original and reformulated queries
                    if 'original_query' in query_data:
                        original_queries[query_id] = query_data['original_query']
                    elif 'org_query_text' in query_data:
                        original_queries[query_id] = query_data['org_query_text']
                    
                    if 'reformulated_query' in query_data:
                        reformulated_queries[query_id] = query_data['reformulated_query']
                    elif 'query_text' in query_data:
                        reformulated_queries[query_id] = query_data['query_text']
                else:
                    # Simple string mapping
                    reformulated_queries[query_id] = str(query_data)
            
            # Retrieve for both original and reformulated queries
            if original_queries:
                orig_output = output_dir / f"run.orig_queries_{subset_name}.trec"
                self.retrieve_batch(original_queries, str(orig_output), f"orig_{subset_name}")
                output_files[f"{subset_name}_original"] = str(orig_output)
            
            if reformulated_queries:
                new_output = output_dir / f"run.new_queries_{subset_name}.trec"
                self.retrieve_batch(reformulated_queries, str(new_output), f"new_{subset_name}")
                output_files[f"{subset_name}_reformulated"] = str(new_output)
        
        return output_files

def create_retriever(retriever_type: str, config: RetrievalConfig) -> Retriever:
    """Factory function to create retrievers"""
    retriever_type = retriever_type.lower()
    
    if retriever_type in ['bm25', 'sparse']:
        return BM25Retriever(config)
    elif retriever_type in ['dense', 'ance', 'tct', 'colbert', 'sbert', 'splade', 'bge']:
        if retriever_type != 'dense' and config.model_name is None:
            config.model_name = retriever_type
        return DenseRetriever(config)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

def main():
    parser = argparse.ArgumentParser(description='General Retrieval Pipeline')
    
    # Input/Output
    parser.add_argument('--input', required=True, help='Input file (TSV or JSON) or dataset directory')
    parser.add_argument('--output', required=True, help='Output file or directory')
    parser.add_argument('--input_format', default='cpu', choices=['auto', 'tsv', 'json', 'dataset'])
    parser.add_argument('--output_format', default='trec', choices=['trec', 'json', 'tsv'])
    
    # Retriever configuration
    parser.add_argument('--retriever_type', required=True, 
                       choices=['bm25', 'sparse', 'dense', 'ance', 'tct', 'colbert', 'sbert', 'splade', 'bge'])
    parser.add_argument('--model_name', help='Model name for dense retrievers')
    parser.add_argument('--encoder_name', help='Encoder name for dense retrievers')
    parser.add_argument('--index_path', help='Path to custom index')
    
    # Retrieval parameters
    parser.add_argument('--hits', type=int, default=1000, help='Number of hits to retrieve')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--parallelism', type=int, default=1, help='Number of parallel threads')
    
    # BM25 parameters
    parser.add_argument('--bm25_k1', type=float, default=0.82, help='BM25 k1 parameter')
    parser.add_argument('--bm25_b', type=float, default=0.68, help='BM25 b parameter')
    
    # Dense retrieval parameters
    parser.add_argument('--ef_search', type=int, default=1000, help='ef_search parameter for HNSW')
    
    # Run configuration
    parser.add_argument('--run_tag', default='retrieval', help='Tag for TREC run')
    parser.add_argument('--subset_names', nargs='*', help='Subset names to process for dataset input')
    
    args = parser.parse_args()
    
    # Create configuration
    config = RetrievalConfig(
        hits=args.hits,
        output_format=args.output_format,
        parallelism=args.parallelism,
        batch_size=args.batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        ef_search=args.ef_search,
        model_name=args.model_name,
        index_path=args.index_path,
        encoder_name=args.encoder_name
    )
    
    # Create retriever
    retriever = create_retriever(args.retriever_type, config)
    pipeline = RetrievalPipeline(retriever, config)
    
    # Determine input format
    input_path = Path(args.input)
    if args.input_format == 'auto':
        if input_path.is_dir():
            input_format = 'dataset'
        elif input_path.suffix == '.json':
            input_format = 'json'
        elif input_path.suffix == '.tsv':
            input_format = 'tsv'
        else:
            raise ValueError(f"Cannot determine input format for {args.input}")
    else:
        input_format = args.input_format
    
    # Process input
    if input_format == 'dataset':
        logger.info("Processing dataset subsets...")
        output_files = pipeline.retrieve_dataset_subsets(
            args.input, 
            args.output, 
            args.subset_names
        )
        logger.info(f"Generated output files: {output_files}")
        
    else:
        # Load queries
        if input_format == 'tsv':
            queries = pipeline.load_queries_from_tsv(args.input)
        elif input_format == 'json':
            queries = pipeline.load_queries_from_json(args.input)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        # Retrieve
        logger.info(f"Starting retrieval for {len(queries)} queries...")
        results = pipeline.retrieve_batch(queries, args.output, args.run_tag)
        logger.info(f"Retrieval complete. Processed {len(results)} queries.")

if __name__ == "__main__":
    main()