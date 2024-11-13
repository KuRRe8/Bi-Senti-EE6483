import pandas as pd
from sklearn.metrics import accuracy_score
import os

df1 = pd.read_csv(os.path.join('data','stripped_labels_keep_reviews_False.csv'))
df2 = pd.read_csv(os.path.join('submit','candidate','distlbert.csv'))

l1 = df1['sentiments'].tolist()
l2 = df2['sentiments'].tolist()

print(accuracy_score(l1, l2))