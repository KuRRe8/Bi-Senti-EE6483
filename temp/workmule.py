import pandas as pd
import json
import os

def convert_json_to_excel():
    data_dir = os.path.join(os.getcwd(), 'Bi-Senti-EE6483/data')
    
    # 转换 train.json
    train_path = os.path.join(data_dir, "train.json")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data)
    train_excel_path = os.path.join(data_dir, "train.xlsx")
    train_df.to_excel(train_excel_path, index=False)
    print(f"Converted {train_path} to {train_excel_path}")
    
    # 转换 test.json
    test_path = os.path.join(data_dir, "test.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data)
    test_excel_path = os.path.join(data_dir, "test.xlsx")
    test_df.to_excel(test_excel_path, index=False)
    print(f"Converted {test_path} to {test_excel_path}")
    
def strip_csv(keep_reviews=False):
    data_dir = os.path.join(os.getcwd(), 'Bi-Senti-EE6483/data')
    print(data_dir)
    labels = os.path.join(data_dir, "labels.csv")
    
    df = pd.read_csv(labels)
    df = df.drop([0, 1])
    if keep_reviews:
        df = df[['reviews', 'compound']]
    else:
        df = df[['compound']]
    
    df[['compound']] = df[['compound']].astype(float)
    df['sentiments'] = df['compound'].apply(lambda x: 1 if x > 0.05 else 0)
    
    if keep_reviews:
        df = df[['reviews', 'sentiments']]
    else:
        df = df[['sentiments']]
        
    stripped_csv_path = os.path.join(data_dir, f"stripped_labels_keep_reviews_{str(keep_reviews)}.csv")
    df.to_csv(stripped_csv_path, index=False)
    print(f"Stripped CSV saved to {stripped_csv_path}")
    
if __name__ == "__main__":
    strip_csv()