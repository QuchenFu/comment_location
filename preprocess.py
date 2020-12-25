import pandas as pd
import pyonmttok
from sklearn.model_selection import train_test_split
import json
tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
df=pd.read_csv('/content/sample_data/data.txt', sep='\t')
quote=[]
score=[]
code=""
comment=0

for index, row in df.iterrows():
  code=code+ str(row[6])+" "
  if row[2]==1:
    comment = 1 if row[3]!=0 else 0
  if row[2]==3:
    quote.append(code[:-1])
    score.append(comment)
    code=""
    comment=0

raw_data = {
    "Quote": quote,
    "Score": score,
}

print(sum(score)/len(score))
new_df = pd.DataFrame(raw_data, columns=["Quote", "Score"])

# create train and test set
train, test = train_test_split(new_df, test_size=0.1, random_state=42)

# Get train, test data to json and csv format which can be read by torchtext
train.to_json("/content/sample_data/train.json", orient="records", lines=True)
test.to_json("/content/sample_data/test.json", orient="records", lines=True)


tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)

def tokenize(text):
    tokens, _ = tokenizer.tokenize(text)
    return tokens