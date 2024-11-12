import os
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split
import optuna
import torch
import pandas

finetuned = os.path.join('checkpoint','bert','final_model')
labeled_data = os.path.join('data','train.json')
unlabeled_data = os.path.join('data','test.json')

if os.path.exists(finetuned):
    
    # read
    df1 = pandas.read_json(labeled_data)
    df2 = pandas.read_json(unlabeled_data)

    df1_500 = df1[:500]
    df2_33 = df2[:33]

    # to dataset
    val_dataset = Dataset.from_dict({
        "text": df1_500["reviews"].tolist(),
        "label": df1_500["sentiments"].tolist()
    })
    pred_dataset = Dataset.from_dict({
        "text": df2_33["reviews"].tolist()
    })


    tokenizer = AutoTokenizer.from_pretrained(finetuned)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned)

    def tokenize_function(examples, tokenizer):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # token
    tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_pred_dataset = pred_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # 设置为评估模式

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in tokenized_val_dataset})
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).cpu().numpy()
        confidence = predictions[range(len(predicted_class)), predicted_class].cpu().numpy()
