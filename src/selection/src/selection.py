import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RefairmulateConfig:
    """Configuration parameters for Refairmulate dataset construction"""
    # Thresholds for categorization
    theta_eff: float = 1.0  # Effectiveness threshold (perfect performance)
    theta_bias: float = 0.0  # Bias threshold (zero bias)
    
    # Scoring function weights (used in build_dataset)
    w_e: float = 1.0  # Weight for effectiveness
    w_b: float = 1.0  # Weight for bias reduction
    
    # Improvement thresholds for labeling
    beta_eff: float = 0.0  # Minimum effectiveness improvement
    beta_bias: float = 0.0  # Minimum bias reduction
    
    # File paths/ Example
    queries_file: str = 'data/msmarco/queries/queries.dev.tsv'
    base_results_file: str = 'data/msmarco/eval/bm25_split_215/full_base.json'
    variant_results_dir: str = 'data/msmarco/eval/bm25_split_215/215'
    variant_queries_file: str = 'data/msmarco/queries/215/queries.train.doct5_215.tsv'
    output_dir: str = '/mnt/data/son/Refairmulate/src/selection/src/data'
    
    # Processing parameters
    num_variants: int = 50
    n_per_group: int = 100


class QueryReader:
    """Handles reading and parsing query files"""
    
    @staticmethod
    def read_queries(file_path: str) -> Dict[str, str]:
        """Read queries from TSV file"""
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1].strip()
        return queries
    
    @staticmethod
    def read_multiple_variant_queries(file_path: str) -> Dict[str, List[str]]:
        """Read multiple query variants from TSV file"""
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Reading variant queries"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    if query_id not in queries:
                        queries[query_id] = []
                    queries[query_id].append(query_text)
        return queries


class ResultsReader:
    """Handles reading evaluation results"""
    
    @staticmethod
    def read_base_results(file_path: str) -> Dict:
        """Read base evaluation results"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    @staticmethod
    def read_multiple_variant_results(results_dir: str, num_variants: int) -> Dict:
        """Read multiple variant evaluation results"""
        results = {}
        for i in tqdm(range(num_variants), desc="Reading variant results"):
            file_path = os.path.join(results_dir, f'full_base_{i}.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    variant_results = json.load(file)
                
                for query_id, metrics in variant_results.items():
                    if query_id not in results:
                        results[query_id] = []
                    results[query_id].append(metrics)
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found")
                continue
        return results


class MetricsEvaluator:
    """Handles metrics evaluation and comparison"""
    
    PERFORMANCE_METRICS = ['RR10', 'AP10']
    BIAS_METRICS = ['LIWC10', 'ARAB-tc10', 'ARAB-tf10', 'ARAB-bool10']
    
    @staticmethod
    def compare_and_label(old_metrics: Dict, new_metrics: Dict) -> Tuple[int, str]:
        """
        Compare old and new metrics to determine improvement label.
        Returns the same 8-category labeling as in original code.
        """
        # Check individual performance metrics (RR10, AP10) - higher is better
        rr10_improved = new_metrics['RR10'] > old_metrics['RR10']
        rr10_equal = new_metrics['RR10'] == old_metrics['RR10']
        rr10_decreased = new_metrics['RR10'] < old_metrics['RR10']
        
        ap10_improved = new_metrics['AP10'] > old_metrics['AP10']
        ap10_equal = new_metrics['AP10'] == old_metrics['AP10']
        ap10_decreased = new_metrics['AP10'] < old_metrics['AP10']
        
        # Check individual bias metrics (LIWC10, ARAB-*) - lower is better
        liwc_improved = new_metrics['LIWC10'] < old_metrics['LIWC10']
        liwc_equal = new_metrics['LIWC10'] == old_metrics['LIWC10']
        liwc_decreased = new_metrics['LIWC10'] > old_metrics['LIWC10']
        
        arab_tc_improved = new_metrics['ARAB-tc10'] < old_metrics['ARAB-tc10']
        arab_tc_equal = new_metrics['ARAB-tc10'] == old_metrics['ARAB-tc10']
        arab_tc_decreased = new_metrics['ARAB-tc10'] > old_metrics['ARAB-tc10']
        
        arab_tf_improved = new_metrics['ARAB-tf10'] < old_metrics['ARAB-tf10']
        arab_tf_equal = new_metrics['ARAB-tf10'] == old_metrics['ARAB-tf10']
        arab_tf_decreased = new_metrics['ARAB-tf10'] > old_metrics['ARAB-tf10']
        
        arab_bool_improved = new_metrics['ARAB-bool10'] < old_metrics['ARAB-bool10']
        arab_bool_equal = new_metrics['ARAB-bool10'] == old_metrics['ARAB-bool10']
        arab_bool_decreased = new_metrics['ARAB-bool10'] > old_metrics['ARAB-bool10']
        
        # Aggregated performance improvements
        all_performance_improved = rr10_improved and ap10_improved
        all_performance_equal = rr10_equal and ap10_equal
        all_performance_decreased = rr10_decreased and ap10_decreased
        
        # Aggregated bias improvements
        all_bias_improved = liwc_improved and arab_tc_improved and arab_tf_improved and arab_bool_improved
        all_bias_equal = liwc_equal and arab_tc_equal and arab_tf_equal and arab_bool_equal
        all_bias_decreased = liwc_decreased and arab_tc_decreased and arab_tf_decreased and arab_bool_decreased
        
        # Determine label
        if all_performance_improved and all_bias_improved:
            return 0, "improve both (all metrics)"
        elif all_bias_improved and all_performance_equal:
            return 1, "improve bias (all metrics), equal performance"
        elif all_performance_improved and all_bias_equal:
            return 2, "improve performance (all metrics), equal bias"
        elif all_performance_equal and all_bias_equal:
            return 3, "no change in any metrics"
        elif all_bias_improved and all_performance_decreased:
            return 4, "improve bias (all metrics), decrease performance (all metrics)"
        elif all_performance_improved and all_bias_decreased:
            return 5, "improve performance (all metrics), decrease bias (all metrics)"
        elif all_performance_decreased and all_bias_decreased:
            return 6, "decrease both (all metrics)"
        else:
            return 7, "mixed results (partial improvements/decreases)"
    
    @staticmethod
    def compute_deltas(old_metrics: Dict, new_metrics: Dict) -> Tuple[float, float]:
        """Compute effectiveness and bias deltas for scoring function"""
        delta_eff = sum(new_metrics[m] - old_metrics[m] 
                       for m in MetricsEvaluator.PERFORMANCE_METRICS)
        delta_bias = sum(old_metrics[m] - new_metrics[m] 
                        for m in MetricsEvaluator.BIAS_METRICS)
        return delta_eff, delta_bias
    
    @staticmethod
    def scoring_function(delta_eff: float, delta_bias: float, w_e: float, w_b: float) -> float:
        """Multi-objective scoring function: S(q, v) = w_e * Δeff + w_b * Δbias"""
        return w_e * delta_eff + w_b * delta_bias


class QueryGrouper:
    """Handles query categorization into groups"""
    
    @staticmethod
    def get_group_statistics(df: pd.DataFrame, config: RefairmulateConfig) -> Dict[int, set]:
        """Get query IDs for each group based on performance and bias thresholds"""
        # Group 1: High Effectiveness, Low Bias (Diamond)
        group_1 = df[((df['RR10'] == config.theta_eff) & (df['AP10'] == config.theta_eff)) &
                     (df['LIWC10'] == config.theta_bias) & 
                     (df['ARAB-tc10'] == config.theta_bias) & 
                     (df['ARAB-tf10'] == config.theta_bias) & 
                     (df['ARAB-bool10'] == config.theta_bias)]

        # Group 2: High Effectiveness, High Bias
        group_2 = df[((df['RR10'] == config.theta_eff) & (df['AP10'] == config.theta_eff)) &
                     ((df['LIWC10'] > config.theta_bias) | 
                      (df['ARAB-tc10'] > config.theta_bias) | 
                      (df['ARAB-tf10'] > config.theta_bias) |
                      (df['ARAB-bool10'] > config.theta_bias))]

        # Group 3: Low Effectiveness, Low Bias
        group_3 = df[((df['RR10'] < config.theta_eff) | (df['AP10'] < config.theta_eff)) &
                     (df['LIWC10'] == config.theta_bias) & 
                     (df['ARAB-tc10'] == config.theta_bias) & 
                     (df['ARAB-tf10'] == config.theta_bias) &
                     (df['ARAB-bool10'] == config.theta_bias)]

        # Group 4: Low Effectiveness, High Bias
        group_4 = df[((df['RR10'] < config.theta_eff) | (df['AP10'] < config.theta_eff)) &
                     ((df['LIWC10'] > config.theta_bias) | 
                      (df['ARAB-tc10'] > config.theta_bias) | 
                      (df['ARAB-tf10'] > config.theta_bias) |
                      (df['ARAB-bool10'] > config.theta_bias))]

        return {
            1: set(group_1.index.tolist()),
            2: set(group_2.index.tolist()),
            3: set(group_3.index.tolist()),
            4: set(group_4.index.tolist())
        }


class QuerySelector:
    """Handles query selection and saves to TSV (no scoring function here)"""
    
    def __init__(self, config: RefairmulateConfig):
        self.config = config
        self.query_reader = QueryReader()
        self.results_reader = ResultsReader()
        self.evaluator = MetricsEvaluator()
        self.grouper = QueryGrouper()
    
    def load_data(self):
        """Load all required data"""
        print("Loading data...")
        
        self.queries_mapping = self.query_reader.read_queries(self.config.queries_file)
        self.variant_queries = self.query_reader.read_multiple_variant_queries(
            self.config.variant_queries_file)
        self.base_results = self.results_reader.read_base_results(self.config.base_results_file)
        self.variant_results = self.results_reader.read_multiple_variant_results(
            self.config.variant_results_dir, self.config.num_variants)
        
        print(f"Loaded {len(self.queries_mapping)} queries")
        print(f"Loaded {len(self.base_results)} base results")
        print(f"Loaded {len(self.variant_results)} variant results")
    
    def process_group(self, group_num: int, query_ids: set) -> Tuple[List[List], Dict]:
        """Process queries for a specific group - just label and save all variants"""
        print(f"\nProcessing Group {group_num}...")
        
        group_pairs = []
        query_labels = defaultdict(list)
        
        for query_id in tqdm(query_ids, desc=f"Group {group_num}"):
            if query_id not in self.variant_results:
                continue
                
            # Process ALL variants, just label them
            for i in range(len(self.variant_results[query_id])):
                if i >= len(self.variant_queries.get(query_id, [])):
                    continue
                    
                variant_text = self.variant_queries[query_id][i]
                old_metrics = self.base_results[query_id]
                new_metrics = self.variant_results[query_id][i]
                
                try:
                    label, label_text = self.evaluator.compare_and_label(old_metrics, new_metrics)
                    
                    # Create pair data with all metrics
                    pair_data = [
                        query_id,
                        variant_text,
                        label,
                        # Original metrics
                        old_metrics['RR10'],
                        old_metrics['AP10'],
                        old_metrics['LIWC10'],
                        old_metrics['ARAB-tc10'],
                        old_metrics['ARAB-tf10'],
                        old_metrics['ARAB-bool10'],
                        # New metrics
                        new_metrics['RR10'],
                        new_metrics['AP10'],
                        new_metrics['LIWC10'],
                        new_metrics['ARAB-tc10'],
                        new_metrics['ARAB-tf10'],
                        new_metrics['ARAB-bool10'],
                        self.queries_mapping[query_id]
                    ]
                    
                    group_pairs.append(pair_data)
                    query_labels[query_id].append(label)
                    
                except Exception as e:
                    print(f"Error processing query ID {query_id}: {e}")
        
        return group_pairs, query_labels
    
    def display_statistics(self, all_groups_data: Dict):
        """Display statistics for each group"""
        label_descriptions = {
            0: "improve both (all metrics)",
            1: "improve bias (all metrics), equal performance",
            2: "improve performance (all metrics), equal bias",
            3: "no change in any metrics",
            4: "improve bias (all metrics), decrease performance (all metrics)",
            5: "improve performance (all metrics), decrease bias (all metrics)",
            6: "decrease both (all metrics)",
            7: "mixed results (partial improvements/decreases)"
        }
        
        for group_num, (group_pairs, query_labels) in all_groups_data.items():
            counts = Counter()
            for query_id, labels in query_labels.items():
                for label in set(labels):
                    counts[label] += 1
            
            total_queries = len(query_labels)
            print(f"\nGroup {group_num} Statistics (Unique Queries: {total_queries})")
            print("-" * 80)
            print(f"{'Label':<5} {'Description':<60} {'Count':<7} {'% of Queries':<10}")
            print("-" * 80)
            
            for label in range(8):
                count = counts.get(label, 0)
                pct = (count / total_queries * 100) if total_queries else 0.0
                desc = label_descriptions.get(label, "Unknown")
                print(f"{label:<5} {desc:<60} {count:<7} {pct:>9.2f}%")
    
    def save_results(self, all_groups_data: Dict):
        """Save results to TSV files"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        headers = [
            'Query ID', 'Query Text', 'Label',
            'Orig_RR10', 'Orig_AP10', 'Orig_LIWC10', 'Orig_ARAB-tc10', 'Orig_ARAB-tf10', 'Orig_ARAB-bool10',
            'New_RR10', 'New_AP10', 'New_LIWC10', 'New_ARAB-tc10', 'New_ARAB-tf10', 'New_ARAB-bool10',
            'Original Query Text'
        ]
        
        for group_num, (group_pairs, _) in all_groups_data.items():
            file_path = os.path.join(self.config.output_dir, f'group_{group_num}_results.tsv')
            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(headers)
                for pair in group_pairs:
                    writer.writerow(pair)
            print(f"Saved {len(group_pairs)} pairs for Group {group_num}")
    
    def run_selection(self):
        """Main selection pipeline - just label and save to TSV"""
        self.load_data()
        
        # Create DataFrame and get groups
        df_all = pd.DataFrame.from_dict(self.base_results, orient='index')
        groups = self.grouper.get_group_statistics(df_all, self.config)
        
        print(f"Group sizes:")
        for group_num, query_ids in groups.items():
            print(f"  Group {group_num}: {len(query_ids)} queries")
        
        # Process each group
        all_groups_data = {}
        for group_num, query_ids in groups.items():
            if query_ids:
                group_pairs, query_labels = self.process_group(group_num, query_ids)
                all_groups_data[group_num] = (group_pairs, query_labels)
        
        # Display statistics
        self.display_statistics(all_groups_data)
        
        # Save results
        self.save_results(all_groups_data)
        
        return all_groups_data


class DatasetBuilder:
    """Builds final datasets using scoring function from TSV files"""
    
    def __init__(self, config: RefairmulateConfig):
        self.config = config
        self.evaluator = MetricsEvaluator()
    
    def collect_metrics_data(self) -> Dict:
        """Collect metrics data from TSV files"""
        metrics_data = {}
        
        for group_num in range(1, 5):
            file_path = os.path.join(self.config.output_dir, f'group_{group_num}_results.tsv')
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                
                for row in reader:
                    query_id = row[0]
                    query_text = row[1]
                    
                    if query_id not in metrics_data:
                        metrics_data[query_id] = {}
                    
                    # Filter based on expected labels for each group
                    label = int(row[2])
                    if (group_num == 1 and label == 3) or \
                       (group_num == 2 and label == 1) or \
                       (group_num == 3 and label == 2) or \
                       (group_num == 4 and label == 0):
                        
                        metrics_data[query_id][query_text] = {
                            'Orig_RR10': float(row[3]),
                            'Orig_AP10': float(row[4]),
                            'Orig_LIWC10': float(row[5]),
                            'Orig_ARAB-tc10': float(row[6]),
                            'Orig_ARAB-tf10': float(row[7]),
                            'Orig_ARAB-bool10': float(row[8]),
                            'New_RR10': float(row[9]),
                            'New_AP10': float(row[10]),
                            'New_LIWC10': float(row[11]),
                            'New_ARAB-tc10': float(row[12]),
                            'New_ARAB-tf10': float(row[13]),
                            'New_ARAB-bool10': float(row[14]),
                            'Label': label,
                            'Group': group_num
                        }
        
        return metrics_data
    
    def select_best_dataset(self, metrics_data: Dict) -> Tuple[Dict, Dict]:
        """Select best queries using scoring function"""
        final_dataset = {1: {}, 2: {}, 3: {}, 4: {}}
        perfect_dataset = {1: {}, 2: {}, 3: {}, 4: {}}
        
        # Group by group_num and label
        group_queries = {
            (1, 3): {},  # Group 1 with label 3 (no change)
            (2, 1): {},  # Group 2 with label 1 (improve bias, equal performance)
            (3, 2): {},  # Group 3 with label 2 (improve performance, equal bias)
            (4, 0): {}   # Group 4 with label 0 (improve both)
        }
        
        # Collect all query variations
        for query_id, variations in metrics_data.items():
            for query_text, metrics in variations.items():
                group = metrics['Group']
                label = metrics['Label']
                key = (group, label)
                
                if key in group_queries:
                    if query_id not in group_queries[key]:
                        group_queries[key][query_id] = []
                    group_queries[key][query_id].append((query_text, metrics))
        
        # Select best queries using scoring function
        queries_mapping = QueryReader.read_queries(self.config.queries_file)
        
        for (group_num, label), queries in group_queries.items():
            query_scores = {}
            perfect_query = {}
            
            for query_id, variations in queries.items():
                best_score = -float('inf')
                best_variation = None
                
                for query_text, metrics in variations:
                    if query_text == queries_mapping.get(query_id, ""):
                        continue
                    
                    # Calculate score using scoring function
                    delta_eff, delta_bias = self.evaluator.compute_deltas(
                        {k.replace('Orig_', ''): v for k, v in metrics.items() if k.startswith('Orig_')},
                        {k.replace('New_', ''): v for k, v in metrics.items() if k.startswith('New_')}
                    )
                    
                    score = self.evaluator.scoring_function(
                        delta_eff, delta_bias, self.config.w_e, self.config.w_b)
                    
                    # Group-specific selection criteria
                    if self._is_valid_for_group(group_num, metrics):
                        if score > best_score:
                            best_score = score
                            best_variation = (query_text, metrics)
                    
                    # Check for perfect queries
                    if self._is_perfect_query(metrics):
                        perfect_query[query_id] = (1, (query_text, metrics))
                
                if best_variation:
                    query_scores[query_id] = (best_score, best_variation[0])
            
            # Add to final dataset
            for query_id, (score, query_text) in query_scores.items():
                final_dataset[group_num][query_id] = {
                    'query_text': query_text,
                    'metrics': metrics_data[query_id][query_text],
                    'score': score,
                    'org_query_text': queries_mapping.get(query_id, "")
                }
            
            # Add perfect queries
            for query_id, (score, (query_text, metrics)) in perfect_query.items():
                perfect_dataset[group_num][query_id] = {
                    'query_text': query_text,
                    'metrics': metrics,
                    'score': score,
                    'org_query_text': queries_mapping.get(query_id, "")
                }
        
        return final_dataset, perfect_dataset
    
    def _is_valid_for_group(self, group_num: int, metrics: Dict) -> bool:
        """Check if variant is valid for specific group"""
        if group_num == 1:
            return True
        elif group_num == 2:
            return any(metrics[f'New_{m}'] > 0 for m in ['ARAB-tc10', 'ARAB-tf10', 'ARAB-bool10'])
        elif group_num == 3:
            return metrics['New_RR10'] < 1 or metrics['New_AP10'] < 1
        elif group_num == 4:
            return not (metrics['New_RR10'] == 1 and metrics['New_AP10'] == 1 and 
                       metrics['New_ARAB-tc10'] == 0 and metrics['New_ARAB-tf10'] == 0 and 
                       metrics['New_ARAB-bool10'] == 0)
        return True
    
    def _is_perfect_query(self, metrics: Dict) -> bool:
        """Check if query achieves perfect performance and bias"""
        return (metrics['New_RR10'] == 1 and metrics['New_AP10'] == 1 and 
                metrics['New_LIWC10'] == 0 and metrics['New_ARAB-tc10'] == 0 and 
                metrics['New_ARAB-tf10'] == 0 and metrics['New_ARAB-bool10'] == 0)
    
    def build_dataset(self) -> Tuple[Dict, Dict]:
        """Build final datasets from TSV files"""
        print("Building datasets from TSV files...")
        
        metrics_data = self.collect_metrics_data()
        final_dataset, perfect_dataset = self.select_best_dataset(metrics_data)
        
        # Save datasets
        improvement_file = os.path.join(self.config.output_dir, '..', 'Effective.json')
        perfect_file = os.path.join(self.config.output_dir, '..', 'Optimal.json')
        
        os.makedirs(os.path.dirname(improvement_file), exist_ok=True)
        
        with open(improvement_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        
        with open(perfect_file, 'w', encoding='utf-8') as f:
            json.dump(perfect_dataset, f, indent=4, ensure_ascii=False)
        
        print(f"Saved improvement dataset to: {improvement_file}")
        print(f"Saved perfect dataset to: {perfect_file}")
        
        return final_dataset, perfect_dataset


def main():
    """Main function"""
    config = RefairmulateConfig()
    
    # Step 1: Run selection (labels and saves to TSV)
    print("=== Step 1: Query Selection ===")
    selector = QuerySelector(config)
    selector.run_selection()
    
    # Step 2: Build dataset (uses scoring function and saves to JSON)
    print("\n=== Step 2: Dataset Building ===")
    builder = DatasetBuilder(config)
    final_dataset, perfect_dataset = builder.build_dataset()
    
    print("\nPipeline completed successfully!")
    return final_dataset, perfect_dataset


if __name__ == "__main__":
    main()