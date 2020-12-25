import pandas as pd
import torch
import spacy
from torchtext.data import Field, TabularDataset, BucketIterator

data_path = '/tmp/pycharm_project_62/split30_word_data/train/data.txt'



def preproces(path):
    df = pd.read_csv(path, sep='\t')
    quote = []
    score = []
    code = ""
    comment = 0

    for index, row in df.iterrows():
        code = code + str(row[6]) + " "
        if row[2] == 1:
            comment = 1 if row[3] != 0 else 0
        if row[2] == 3:
            quote.append(code[:-1])
            score.append(comment)
            code = ""
            comment = 0

    data = {
        "Quote": quote,
        "Score": score,
    }

    data = pd.DataFrame(data, columns=["Quote", "Score"])
    data.to_json("data.json", orient="records", lines=True)

preproces(data_path)


device = torch.device("cpu")

# python -m spacy download en
spacy_en = spacy.load("en")


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {"Quote": ("q", quote), "Score": ("s", score)}

train_data, test_data = TabularDataset.splits(
    path="", train="data.json", test="data.json", format="json", fields=fields
)

quote.build_vocab(train_data, max_size=10000, min_freq=10, vectors="glove.6B.100d")

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=1, device=device,sort=False
)

for batch_idx, batch in enumerate(train_iterator):
    # Get data to cuda if possible
    data = batch.q.to(device=device)
    targets = batch.s.to(device=device)
    print(data)
    print(targets)

#
# tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
#
# def tokenize(text):
#     tokens, _ = tokenizer.tokenize(text)
#     return tokens
