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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


naame = 'roberta'

prefix = '/home/users/ntu/yang0886/proj/Bi-Senti-EE6483/'
finetuned = os.path.join(prefix,'checkpoint',naame,'final_model')
labeled_data = os.path.join(prefix,'data','train.json')
unlabeled_data = os.path.join(prefix,'data','test.json')

if os.path.exists(finetuned):
    
    # read
    df1 = pandas.read_json(labeled_data)
    df2 = pandas.read_json(unlabeled_data)

    df1_500 = df1
    df2_33 = df2

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
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text","label"])
    tokenized_pred_dataset = tokenized_pred_dataset.remove_columns(["text"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # 设置为评估模式

    with torch.no_grad():

        output_val = []
        for i in range(len(tokenized_val_dataset)):
            current_input = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in tokenized_val_dataset[i].items()}
            outputs = model(**current_input)
            prediction_logits = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction_class = torch.argmax(prediction_logits, dim=-1).item()
            output_val.append(prediction_class)
        
        print('val acc:',accuracy_score(df1_500["sentiments"].tolist(),output_val))

        output_pred = []
        for i in range(len(tokenized_pred_dataset)):
            current_input = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in tokenized_pred_dataset[i].items()}
            outputs = model(**current_input)
            prediction_logits = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction_class = torch.argmax(prediction_logits, dim=-1).item()
            output_pred.append(prediction_class)
        
        s = pandas.Series(output_pred)
        df2['sentiments'] = s
        df2.to_csv(f'{naame}.csv',index=False)
