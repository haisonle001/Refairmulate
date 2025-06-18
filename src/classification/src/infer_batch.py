import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_dir, device):
    """Load the tokenizer and model from the specified directory and move the model to the given device."""
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)  # Move model to GPU if available
    model.eval()
    return tokenizer, model

def predict_batch(texts, model, tokenizer, device):
    """Predict labels for a batch of texts using GPU if available."""
    encoding = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(device)  # Move tensors to GPU if available

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()  # Move back to CPU for processing
    
    return predictions

def process_file(input_file, output_file, model, tokenizer, device, batch_size=64):
    """Process an input file in batches and save predictions to an output file with a progress bar."""
    df = pd.read_csv(input_file, sep="\t", header=None, names=["id", "text", "label"])

    texts = df["text"].tolist()
    predictions = []

    # Process in batches with tqdm progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches", unit="batch"):
        batch_texts = texts[i:i+batch_size]  # Get batch
        batch_predictions = predict_batch(batch_texts, model, tokenizer, device)
        predictions.extend(batch_predictions)

    df["predicted_label"] = predictions
    df.to_csv(output_file, sep="\t", index=False)

    print(f"✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file (TSV format)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file (TSV format)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing (default: 64)")
    args = parser.parse_args()

    # Detect GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    tokenizer, model = load_model(args.model_dir, device)
    process_file(args.input_file, args.output_file, model, tokenizer, device, args.batch_size)
