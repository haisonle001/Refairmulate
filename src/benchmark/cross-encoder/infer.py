import pickle
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
import pandas as pd
import json


def predict_query_similarity(dataset_path, trained_model_path, output_path):
    # Read the TSV dataset
    # dataset = pd.read_csv(dataset_path, sep='\t')
    dataset= json.load(open(dataset_path, 'r', encoding='utf-8'))
    
    # Initialize the CrossEncoder model
    model = CrossEncoder(trained_model_path, num_labels=1)
    
    # Prepare input examples for prediction
    sentences = []
    for query_id,  variations in dataset.items():
        for variation in variations:
            sentences.append([f"{variation['orig_query']}[SEP]{variation['orig_document']}", f"{variation['new_query']}[SEP]{variation['new_document']}"])
            
    # f'{variation['orig_query']}[SEP] Retrieved DOC'

    
    # Predict similarity scores
    scores = model.predict(sentences)
    print(f"Predicted {len(scores)} similarity scores. {scores}")
    i=0
    # Add scores to the dataset
    for query_id, variations in dataset.items():
        for variation in variations:
            variation['score'] = float(scores[i])
            i+=1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)    
    
    return dataset

# Example usage
if __name__ == "__main__":
    trained_model_path = "benchmark/cross-encoder/bm25_model"
    dataset_path = "/mnt/data/son/Thesis/data/msmarco/eval/bm25_split_1765/variations.json"
    output_path = "/mnt/data/son/Thesis/classification.archive.models/1765/1765_bm25.json"
    
    result_df = predict_query_similarity(dataset_path, trained_model_path, output_path)