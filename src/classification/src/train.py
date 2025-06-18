import torch
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Define dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["id", "text", "label"])
    label_map = {label: idx for idx, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label_map)
    return df, label_map

# Train function
def train_model(train_file, model_name="bert-base-uncased", output_dir="bert_model"):
    # Load and split dataset (90% train, 10% validation)
    df, label_map = load_data(train_file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label"], test_size=0.1, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = CustomDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = CustomDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    args = parser.parse_args()
    train_model(args.train_file)
