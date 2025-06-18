import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import itertools
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    # Cutoff ranks for evaluation
    cutoffs: List[int] = None
    
    # Resource paths
    liwc_dict_path: Optional[str] = None
    docs_bias_paths: Optional[Dict[str, str]] = None
    collection_neutrality_path: Optional[str] = None
    qrels_path: Optional[str] = None
    
    # Metrics to compute
    compute_retrieval_metrics: bool = True
    compute_liwc: bool = True
    compute_arab: bool = True
    compute_fairr: bool = True
    
    # Output configuration
    output_format: str = 'json'  # 'json', 'tsv', 'both'
    save_per_query: bool = True
    save_aggregated: bool = True
    
    def __post_init__(self):
        if self.cutoffs is None:
            self.cutoffs = [10, 20]
        
        # Set default resource paths if not provided
        if self.liwc_dict_path is None:
            self.liwc_dict_path = "./resources/liwccollection_bias.pkl"
        
        if self.docs_bias_paths is None:
            self.docs_bias_paths = {
                'tc': "./resources/msmarco_passage_docs_bias_tc.pkl",
                'tf': "./resources/msmarco_passage_docs_bias_tf.pkl",
                'bool': "./resources/msmarco_passage_docs_bias_bool.pkl",
            }
        
        if self.collection_neutrality_path is None:
            self.collection_neutrality_path = "./resources/outputcollection_neutralityscores.tsv"
        
        if self.qrels_path is None:
            self.qrels_path = "./data/qrels.train.tsv"


class RetrievalMetricsCalculator:
    """Calculator for retrieval effectiveness metrics (RR, AP)"""
    
    @staticmethod
    def compute_rr_for_query(qrel_for_query: Dict[str, int], run_list: List[str], k: int) -> float:
        """Compute reciprocal rank (RR) for a single query with cutoff k."""
        for i, docid in enumerate(run_list[:k]):
            if docid in qrel_for_query and qrel_for_query[docid] > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def compute_ap_for_query(qrel_for_query: Dict[str, int], run_list: List[str], k: int) -> float:
        """Compute average precision (AP) for a single query with cutoff k."""
        relevant_docs = {docid for docid, rel in qrel_for_query.items() if rel > 0}
        if not relevant_docs:
            return 0.0
        
        num_relevant = 0
        ap = 0.0
        for i, docid in enumerate(run_list[:k]):
            if docid in relevant_docs:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        return ap / len(relevant_docs)
    
    def compute_metrics_per_query(self, qrel: Dict, run: Dict, cutoffs: List[int]) -> Dict:
        """Return metrics for each query at specified cutoffs."""
        per_query_results = {}
        
        for qid, run_list in run.items():
            per_query_results[qid] = {}
            
            if qid not in qrel:
                logger.warning(f"Query ID {qid} not found in qrels.")
                for k in cutoffs:
                    per_query_results[qid][f"RR{k}"] = 0.0
                    per_query_results[qid][f"AP{k}"] = 0.0
                continue
            
            for k in cutoffs:
                rr = self.compute_rr_for_query(qrel[qid], run_list, k)
                ap = self.compute_ap_for_query(qrel[qid], run_list, k)
                per_query_results[qid][f"RR{k}"] = rr
                per_query_results[qid][f"AP{k}"] = ap
        
        return per_query_results


class LIWCBiasCalculator:
    """Calculator for LIWC bias scores"""
    
    def __init__(self, liwc_dict_path: str):
        self.liwc_dict_path = liwc_dict_path
        self.liwc_dict = None
        self.load_liwc_dict()
    
    def load_liwc_dict(self):
        """Load LIWC bias dictionary"""
        try:
            with open(self.liwc_dict_path, "rb") as f:
                self.liwc_dict = pickle.load(f)
            logger.info(f"Loaded LIWC dictionary with {len(self.liwc_dict)} documents")
        except FileNotFoundError:
            logger.error(f"LIWC dictionary not found: {self.liwc_dict_path}")
            self.liwc_dict = {}
        except Exception as e:
            logger.error(f"Error loading LIWC dictionary: {e}")
            self.liwc_dict = {}
    
    def find_top_n_docs(self, run_data: Dict[str, List[str]], cutoff: int) -> Dict[str, List[str]]:
        """Extract top-n documents for each query from run data."""
        top_n_docs = {}
        for query_id, doc_list in run_data.items():
            top_n_docs[query_id] = doc_list[:cutoff]
        return top_n_docs
    
    def calculate_query_score_cutoff(self, doc_ids: List[str]) -> List[float]:
        """Calculate the mean LIWC score for a query based on its top-n docs."""
        if not doc_ids:
            return [0.0, 0.0, 0.0]
        
        query_score = [0.0, 0.0, 0.0]
        valid_docs = 0
        
        for doc_id in doc_ids:
            try:
                key = int(doc_id)
            except ValueError:
                key = doc_id
            
            if key in self.liwc_dict:
                scores = self.liwc_dict[key]
                query_score = [query_score[i] + scores[i] for i in range(3)]
                valid_docs += 1
        
        if valid_docs > 0:
            query_score = [abs(score / valid_docs) * 100 for score in query_score]
        
        return query_score
    
    def calculate_liwc_scores(self, run_data: Dict[str, List[str]], cutoffs: List[int]) -> Dict:
        """Calculate LIWC scores for all queries at specified cutoffs."""
        results = {}
        
        for cutoff in cutoffs:
            top_n_docs = self.find_top_n_docs(run_data, cutoff)
            cutoff_results = {}
            
            for query_id, doc_ids in top_n_docs.items():
                scores = self.calculate_query_score_cutoff(doc_ids)
                cutoff_results[query_id] = {
                    f"LIWC{cutoff}": round(scores[2], 4)  # Use the third score (index 2)
                }
            
            results[cutoff] = cutoff_results
        
        return results


class ARaBCalculator:
    """Calculator for ARaB (Average Ranking-aware Bias) scores"""
    
    def __init__(self, docs_bias_paths: Dict[str, str]):
        self.docs_bias_paths = docs_bias_paths
        self.docs_bias = {}
        self.load_docs_bias()
    
    def load_docs_bias(self):
        """Load document bias values for all methods."""
        for method, path in self.docs_bias_paths.items():
            try:
                with open(path, 'rb') as f:
                    self.docs_bias[method] = pickle.load(f)
                logger.info(f"Loaded {method} bias data with {len(self.docs_bias[method])} documents")
            except FileNotFoundError:
                logger.error(f"Bias file not found: {path}")
                self.docs_bias[method] = {}
            except Exception as e:
                logger.error(f"Error loading bias file {path}: {e}")
                self.docs_bias[method] = {}
    
    @staticmethod
    def calc_RaB_q(bias_list: List[Tuple], at_rank: int) -> Tuple[float, float, float]:
        """Calculate Ranking-aware Bias for a query at a specific rank."""
        if not bias_list or at_rank <= 0:
            return 0.0, 0.0, 0.0
        
        relevant_bias = bias_list[:at_rank]
        bias_val = np.mean([x[0] for x in relevant_bias])
        bias_feml_val = np.mean([x[1] for x in relevant_bias])
        bias_male_val = np.mean([x[2] for x in relevant_bias])
        return bias_val, bias_feml_val, bias_male_val
    
    @staticmethod
    def calc_ARaB_q(bias_list: List[Tuple], at_rank: int) -> Tuple[float, float, float]:
        """Calculate Average Ranking-aware Bias for a query up to a specific rank."""
        if not bias_list or at_rank <= 0:
            return 0.0, 0.0, 0.0
        
        _vals = []
        _feml_vals = []
        _male_vals = []
        
        for t in range(min(at_rank, len(bias_list))):
            _val_RaB, _feml_val_RaB, _male_val_RaB = ARaBCalculator.calc_RaB_q(bias_list, t + 1)
            _vals.append(abs(_val_RaB))
            _feml_vals.append(abs(_feml_val_RaB))
            _male_vals.append(abs(_male_val_RaB))
        
        if _vals:
            bias_val = np.mean(_vals)
            bias_feml_val = np.mean(_feml_vals)
            bias_male_val = np.mean(_male_vals)
        else:
            bias_val = bias_feml_val = bias_male_val = 0.0
        
        return bias_val, bias_feml_val, bias_male_val
    
    def process_run_for_arab(self, run_data: Dict[str, List[str]]) -> Dict:
        """Process run data to prepare for ARaB calculation."""
        runs_docs_bias = {}
        
        for method in self.docs_bias:
            runs_docs_bias[method] = {}
            
            for query_id, doc_list in run_data.items():
                runs_docs_bias[method][query_id] = []
                
                for doc_id in doc_list:
                    try:
                        doc_id_int = int(doc_id)
                    except ValueError:
                        continue
                    
                    if doc_id_int in self.docs_bias[method]:
                        runs_docs_bias[method][query_id].append(self.docs_bias[method][doc_id_int])
        
        return runs_docs_bias
    
    def compute_arab_scores(self, run_data: Dict[str, List[str]], cutoffs: List[int]) -> Dict:
        """Compute ARaB scores for all queries at specified cutoffs."""
        runs_docs_bias = self.process_run_for_arab(run_data)
        qry_bias_ARaB = {}
        
        for method in self.docs_bias:
            qry_bias_ARaB[method] = {}
            
            for at_rank in cutoffs:
                qry_bias_ARaB[method][at_rank] = {}
                
                for qry_id in runs_docs_bias[method]:
                    if runs_docs_bias[method][qry_id]:
                        arab_scores = self.calc_ARaB_q(runs_docs_bias[method][qry_id], at_rank)
                        qry_bias_ARaB[method][at_rank][qry_id] = arab_scores
        
        return qry_bias_ARaB


class FaiRRCalculator:
    """Calculator for FaiRR and NFaiRR scores"""
    
    def __init__(self, collection_neutrality_path: str):
        self.collection_neutrality_path = collection_neutrality_path
        self.documents_neutrality = {}
        self.position_biases = [1/(np.log2(rank+1)) for rank in range(1, 1001)]
        self.load_neutrality_scores()
    
    def load_neutrality_scores(self):
        """Load document neutrality scores."""
        try:
            with open(self.collection_neutrality_path, 'r') as f:
                for line in f:
                    vals = line.strip().split('\t')
                    if len(vals) >= 2:
                        try:
                            doc_id = int(vals[0])
                            neutrality = float(vals[1])
                            self.documents_neutrality[doc_id] = neutrality
                        except ValueError:
                            continue
            logger.info(f"Loaded neutrality scores for {len(self.documents_neutrality)} documents")
        except FileNotFoundError:
            logger.error(f"Neutrality file not found: {self.collection_neutrality_path}")
        except Exception as e:
            logger.error(f"Error loading neutrality file: {e}")
    
    def calculate_fairr_scores(self, 
                              run_data: Dict[str, List[str]], 
                              background_data: Dict[str, List[str]], 
                              cutoffs: List[int]) -> Dict:
        """Calculate FaiRR and NFaiRR scores."""
        # Calculate ideal FaiRR for background
        ideal_fairr = self._calculate_ideal_fairr(background_data, cutoffs)
        
        # Calculate actual FaiRR for run
        actual_fairr = self._calculate_actual_fairr(run_data, cutoffs)
        
        # Calculate NFaiRR (normalized)
        nfairr_results = {}
        for cutoff in cutoffs:
            nfairr_results[cutoff] = {}
            for qid in actual_fairr.get(cutoff, {}):
                if qid in ideal_fairr.get(cutoff, {}) and ideal_fairr[cutoff][qid] > 0:
                    nfairr_results[cutoff][qid] = actual_fairr[cutoff][qid] / ideal_fairr[cutoff][qid]
        
        return {
            'FaiRR': actual_fairr,
            'NFaiRR': nfairr_results
        }
    
    def _calculate_ideal_fairr(self, background_data: Dict[str, List[str]], cutoffs: List[int]) -> Dict:
        """Calculate ideal FaiRR scores for background data."""
        ideal_scores = {}
        
        for cutoff in cutoffs:
            ideal_scores[cutoff] = {}
            
            for qid, doc_list in background_data.items():
                # Get neutrality scores for documents
                neutrality_scores = []
                for doc_id in doc_list:
                    try:
                        doc_id_int = int(doc_id)
                        if doc_id_int in self.documents_neutrality:
                            neutrality_scores.append(self.documents_neutrality[doc_id_int])
                        else:
                            neutrality_scores.append(1.0)  # Default neutrality
                    except ValueError:
                        neutrality_scores.append(1.0)
                
                # Sort in descending order for ideal ranking
                neutrality_scores.sort(reverse=True)
                
                # Calculate ideal FaiRR
                cutoff_len = min(len(neutrality_scores), cutoff)
                if cutoff_len > 0:
                    ideal_score = np.sum(np.multiply(
                        neutrality_scores[:cutoff_len], 
                        self.position_biases[:cutoff_len]
                    ))
                    ideal_scores[cutoff][qid] = ideal_score
        
        return ideal_scores
    
    def _calculate_actual_fairr(self, run_data: Dict[str, List[str]], cutoffs: List[int]) -> Dict:
        """Calculate actual FaiRR scores for run data."""
        actual_scores = {}
        
        for cutoff in cutoffs:
            actual_scores[cutoff] = {}
            
            for qid, doc_list in run_data.items():
                # Get neutrality scores in ranking order
                neutrality_scores = []
                for doc_id in doc_list[:cutoff]:
                    try:
                        doc_id_int = int(doc_id)
                        if doc_id_int in self.documents_neutrality:
                            neutrality_scores.append(self.documents_neutrality[doc_id_int])
                        else:
                            neutrality_scores.append(1.0)  # Default neutrality
                    except ValueError:
                        neutrality_scores.append(1.0)
                
                # Calculate actual FaiRR
                if neutrality_scores:
                    actual_score = np.sum(np.multiply(
                        neutrality_scores, 
                        self.position_biases[:len(neutrality_scores)]
                    ))
                    actual_scores[cutoff][qid] = actual_score
        
        return actual_scores


class EvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.retrieval_calc = RetrievalMetricsCalculator() if config.compute_retrieval_metrics else None
        self.liwc_calc = LIWCBiasCalculator(config.liwc_dict_path) if config.compute_liwc else None
        self.arab_calc = ARaBCalculator(config.docs_bias_paths) if config.compute_arab else None
        self.fairr_calc = FaiRRCalculator(config.collection_neutrality_path) if config.compute_fairr else None
    
    def load_qrels(self, qrels_path: str) -> Dict:
        """Load qrels file."""
        qrel = {}
        try:
            with open(qrels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid, _, did, label = parts[:4]
                        if qid not in qrel:
                            qrel[qid] = {}
                        qrel[qid][did] = int(label)
            logger.info(f"Loaded qrels for {len(qrel)} queries")
        except FileNotFoundError:
            logger.error(f"Qrels file not found: {qrels_path}")
        except Exception as e:
            logger.error(f"Error loading qrels: {e}")
        return qrel
    
    def load_run_from_trec(self, trec_file: str) -> Dict[str, List[str]]:
        """Load run data from TREC format file."""
        run = {}
        try:
            with open(trec_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        qid, _, did, _, _, _ = parts[:6]
                        if qid not in run:
                            run[qid] = []
                        run[qid].append(did)
            logger.info(f"Loaded run data for {len(run)} queries")
        except FileNotFoundError:
            logger.error(f"Run file not found: {trec_file}")
        except Exception as e:
            logger.error(f"Error loading run file: {e}")
        return run
    
    def load_run_from_json(self, json_file: str) -> Dict[str, List[str]]:
        """Load run data from JSON format file."""
        run = {}
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for qid, results in data.items():
                if isinstance(results, list):
                    # Extract docids from list of result dictionaries
                    run[qid] = [r['docid'] if isinstance(r, dict) else str(r) for r in results]
                else:
                    logger.warning(f"Unexpected format for query {qid}")
            
            logger.info(f"Loaded run data for {len(run)} queries")
        except FileNotFoundError:
            logger.error(f"Run file not found: {json_file}")
        except Exception as e:
            logger.error(f"Error loading run file: {e}")
        return run
    
    def evaluate_single_run(self, 
                          run_file: str, 
                          qrels_file: Optional[str] = None,
                          background_run_file: Optional[str] = None) -> Dict:
        """Evaluate a single run file."""
        # Load run data
        if run_file.endswith('.json'):
            run_data = self.load_run_from_json(run_file)
        else:
            run_data = self.load_run_from_trec(run_file)
        
        if not run_data:
            logger.error("No run data loaded")
            return {}
        
        # Load qrels
        qrels_path = qrels_file or self.config.qrels_path
        qrel = self.load_qrels(qrels_path) if self.config.compute_retrieval_metrics else {}
        
        # Initialize results
        combined_results = {}
        
        # Initialize per-query results for each query
        for qid in run_data.keys():
            combined_results[qid] = {}
        
        # Compute retrieval metrics
        if self.config.compute_retrieval_metrics and self.retrieval_calc:
            logger.info("Computing retrieval metrics...")
            retrieval_results = self.retrieval_calc.compute_metrics_per_query(
                qrel, run_data, self.config.cutoffs
            )
            
            for qid in combined_results:
                if qid in retrieval_results:
                    combined_results[qid].update(retrieval_results[qid])
        
        # Compute LIWC scores
        if self.config.compute_liwc and self.liwc_calc:
            logger.info("Computing LIWC scores...")
            liwc_results = self.liwc_calc.calculate_liwc_scores(run_data, self.config.cutoffs)
            
            for cutoff in self.config.cutoffs:
                if cutoff in liwc_results:
                    for qid in combined_results:
                        if qid in liwc_results[cutoff]:
                            combined_results[qid].update(liwc_results[cutoff][qid])
        
        # Compute ARaB scores
        if self.config.compute_arab and self.arab_calc:
            logger.info("Computing ARaB scores...")
            arab_results = self.arab_calc.compute_arab_scores(run_data, self.config.cutoffs)
            
            for method in arab_results:
                for cutoff in self.config.cutoffs:
                    if cutoff in arab_results[method]:
                        for qid in combined_results:
                            if qid in arab_results[method][cutoff]:
                                score = arab_results[method][cutoff][qid][0]  # Use first score
                                combined_results[qid][f"ARAB-{method}{cutoff}"] = round(score, 4)
        
        # Compute FaiRR scores
        if self.config.compute_fairr and self.fairr_calc:
            logger.info("Computing FaiRR scores...")
            
            # Use background run or current run as background
            background_data = run_data
            if background_run_file:
                if background_run_file.endswith('.json'):
                    background_data = self.load_run_from_json(background_run_file)
                else:
                    background_data = self.load_run_from_trec(background_run_file)
            
            fairr_results = self.fairr_calc.calculate_fairr_scores(
                run_data, background_data, self.config.cutoffs
            )
            
            for metric_type in ['FaiRR', 'NFaiRR']:
                if metric_type in fairr_results:
                    for cutoff in self.config.cutoffs:
                        if cutoff in fairr_results[metric_type]:
                            for qid in combined_results:
                                if qid in fairr_results[metric_type][cutoff]:
                                    score = fairr_results[metric_type][cutoff][qid]
                                    combined_results[qid][f"{metric_type}{cutoff}"] = round(score, 4)
        
        return combined_results
    
    def evaluate_multiple_runs(self, 
                             run_files: List[str], 
                             output_dir: str,
                             qrels_file: Optional[str] = None,
                             background_run_file: Optional[str] = None) -> Dict[str, Dict]:
        """Evaluate multiple run files."""
        all_results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for run_file in tqdm(run_files, desc="Evaluating runs"):
            logger.info(f"Evaluating {run_file}")
            
            # Evaluate single run
            results = self.evaluate_single_run(run_file, qrels_file, background_run_file)
            
            # Generate output filename
            run_name = Path(run_file).stem
            
            # Save individual results
            if self.config.save_per_query:
                self.save_results(results, output_dir / f"{run_name}_per_query", per_query=True)
            
            if self.config.save_aggregated:
                self.save_results(results, output_dir / f"{run_name}_aggregated", per_query=False)
            
            all_results[run_name] = results
        
        return all_results
    
    def save_results(self, results: Dict, output_path: Path, per_query: bool = True):
        """Save evaluation results."""
        if per_query:
            # Save per-query results
            if self.config.output_format in ['json', 'both']:
                with open(f"{output_path}.json", 'w') as f:
                    json.dump(results, f, indent=4)
            
            if self.config.output_format in ['tsv', 'both']:
                self._save_tsv(results, f"{output_path}.tsv", per_query=True)
        
        else:
            # Save aggregated results
            aggregated = self._compute_aggregated_metrics(results)
            
            if self.config.output_format in ['json', 'both']:
                with open(f"{output_path}.json", 'w') as f:
                    json.dump(aggregated, f, indent=4)
            
            if self.config.output_format in ['tsv', 'both']:
                self._save_tsv(aggregated, f"{output_path}.tsv", per_query=False)
    
    def _save_tsv(self, results: Dict, output_path: str, per_query: bool = True):
        """Save results in TSV format."""
        if not results:
            return
        
        if per_query:
            # Save per-query results
            with open(output_path, 'w') as f:
                # Get all metric names
                all_metrics = set()
                for qid_results in results.values():
                    all_metrics.update(qid_results.keys())
                
                metrics = sorted(all_metrics)
                
                # Write header
                f.write("query_id\t" + "\t".join(metrics) + "\n")
                
                # Write data
                for qid, qid_results in results.items():
                    row = [qid]
                    for metric in metrics:
                        value = qid_results.get(metric, "")
                        row.append(str(value))
                    f.write("\t".join(row) + "\n")
        else:
            # Save aggregated results
            with open(output_path, 'w') as f:
                f.write("metric\tvalue\n")
                for metric, value in results.items():
                    f.write(f"{metric}\t{value}\n")
    
    def _compute_aggregated_metrics(self, results: Dict) -> Dict:
        """Compute aggregated metrics from per-query results."""
        if not results:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for qid_results in results.values():
            all_metrics.update(qid_results.keys())
        
        aggregated = {}
        query_count = len(results)
        
        for metric in all_metrics:
            values = []
            for qid_results in results.values():
                if metric in qid_results and qid_results[metric] is not None:
                    try:
                        values.append(float(qid_results[metric]))
                    except (ValueError, TypeError):
                        continue
            
            if values:
                avg_value = sum(values) / len(values)
                aggregated[f"avg_{metric}"] = round(avg_value, 5)
        
        return aggregated


def main():
    parser = argparse.ArgumentParser(description='General Evaluation Pipeline')
    
    # Input/Output
    parser.add_argument('--runs', nargs='+', required=True, help='Run files to evaluate')
    parser.add_argument('--qrels', help='Path to qrels file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--background_run', help='Background run file for FaiRR')
    
    # Evaluation configuration
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[10, 20], help='Cutoff ranks')
    parser.add_argument('--output_format', default='json', choices=['json', 'tsv', 'both'])
    parser.add_argument('--save_per_query', action='store_true', default=True)
    parser.add_argument('--save_aggregated', action='store_true', default=True)
    
    # Metrics to compute
    parser.add_argument('--compute_retrieval', action='store_true', default=True)
    parser.add_argument('--compute_liwc', action='store_true', default=True)
    parser.add_argument('--compute_arab', action='store_true', default=True)
    parser.add_argument('--compute_fairr', action='store_true', default=True)
    
    # Resource paths
    parser.add_argument('--liwc_dict_path', default='./resources/liwccollection_bias.pkl')
    parser.add_argument('--docs_bias_tc_path', default='./resources/msmarco_passage_docs_bias_tc.pkl')
    parser.add_argument('--docs_bias_tf_path', default='./resources/msmarco_passage_docs_bias_tf.pkl')
    parser.add_argument('--docs_bias_bool_path', default='./resources/msmarco_passage_docs_bias_bool.pkl')
    parser.add_argument('--collection_neutrality_path', default='./resources/outputcollection_neutralityscores.tsv')
    
    args = parser.parse_args()
    
    # Create docs_bias_paths dictionary
    docs_bias_paths = {
        'tc': args.docs_bias_tc_path,
        'tf': args.docs_bias_tf_path,
        'bool': args.docs_bias_bool_path
    }
    
    # Create configuration
    config = EvaluationConfig(
        cutoffs=args.cutoffs,
        liwc_dict_path=args.liwc_dict_path,
        docs_bias_paths=docs_bias_paths,
        collection_neutrality_path=args.collection_neutrality_path,
        qrels_path=args.qrels,
        compute_retrieval_metrics=args.compute_retrieval,
        compute_liwc=args.compute_liwc,
        compute_arab=args.compute_arab,
        compute_fairr=args.compute_fairr,
        output_format=args.output_format,
        save_per_query=args.save_per_query,
        save_aggregated=args.save_aggregated
    )
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    # Evaluate runs
    logger.info(f"Starting evaluation of {len(args.runs)} run file(s)")
    results = pipeline.evaluate_multiple_runs(
        args.runs,
        args.output_dir,
        args.qrels,
        args.background_run
    )
    
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Print summary statistics
    for run_name, run_results in results.items():
        aggregated = pipeline._compute_aggregated_metrics(run_results)
        logger.info(f"\nSummary for {run_name}:")
        
        # Print key metrics
        key_metrics = []
        for cutoff in config.cutoffs:
            key_metrics.extend([f"avg_RR{cutoff}", f"avg_AP{cutoff}", f"avg_LIWC{cutoff}"])
            for method in ['tc', 'tf', 'bool']:
                key_metrics.append(f"avg_ARAB-{method}{cutoff}")
        
        for metric in key_metrics:
            if metric in aggregated:
                logger.info(f"  {metric}: {aggregated[metric]:.5f}")


if __name__ == "__main__":
    main()