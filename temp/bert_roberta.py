import os
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split
import optuna
import torch

# 设置路径
_root_path_ = 'D:\\_work\\Bi-Senti-EE6483'
if '_root_path_' in locals():
    os.chdir(_root_path_)
assert os.path.basename(os.getcwd()) == 'Bi-Senti-EE6483'

# 加载数据
def load_custom_dataset(train_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item["reviews"] for item in data]
    labels = [item["sentiments"] for item in data]
    
    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })
    
    test_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_labels
    })
    
    return train_dataset, test_dataset

# 数据预处理
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 定义评估函数
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 定义超参数优化的目标函数
def objective(trial: optuna.Trial):
    # 定义超参数搜索空间
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
    }
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join('log', f"trial_roberta_{trial.number}"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=hyperparameters["learning_rate"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        weight_decay=hyperparameters["weight_decay"],
        load_best_model_at_end=True
    )
    
    # 初始化模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", 
        num_labels=2
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 训练模型
    trainer.train()
    
    # 评估模型
    eval_result = trainer.evaluate()
    
    return eval_result["eval_accuracy"]

def main():
    # 1. 加载数据
    print("1. 加载数据")
    train_dataset, test_dataset = load_custom_dataset(os.path.join('data', 'train.json'))
    
    # 2. 初始化tokenizer
    print("2. 初始化tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")    

    # 3. 数据预处理
    print("3. 数据预处理")
    global tokenized_train_dataset, tokenized_test_dataset
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    
    # 4. 超参数优化
    print("4. 超参数优化")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    # 5. 使用最佳超参数训练最终模型
    print("5. 使用最佳超参数训练最终模型")
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {study.best_value}")
    
    # 6. 使用最佳参数训练最终模型
    print("6. 使用最佳参数训练最终模型")
    final_training_args = TrainingArguments(
        output_dir=os.path.join('checkpoint', 'roberta'),
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )
    
    final_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", 
        num_labels=2
    )
    
    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 训练最终模型
    print("训练最终模型")
    final_trainer.train()
    
    # 评估最终模型
    print("评估最终模型")
    final_eval_result = final_trainer.evaluate()
    print(f"Final test accuracy: {final_eval_result['eval_accuracy']}")
    
    # 保存最终模型
    print("保存最终模型")
    final_model_path = os.path.join('checkpoint', 'roberta', 'final_model')
    final_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

if __name__ == "__main__":
    main()