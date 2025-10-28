import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)  # Move the model to the GPU (if available)
    model.eval()
    return tokenizer, model

def predict(text, model, tokenizer):
    device = next(model.parameters()).device  # Get the device of the model
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=64
    ).to(device)  # Move the encoding to the same device as the model

    with torch.no_grad():
        outputs = model(**encoding)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    label = predict(args.text, model, tokenizer)
    print(f"Predicted Label: {label}")
