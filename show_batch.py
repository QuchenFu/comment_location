import pandas as pd
import torch
import argparse
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import pyonmttok

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='split30_word_data/train/data.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser

def preprocess(path):
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


#
# def tokenize(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]
tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)

def tokenize(text):
    tokens, _ = tokenizer.tokenize(text)
    return tokens

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cpu")
    preprocess(args.filepath)
    quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
    score = Field(sequential=False, use_vocab=False)

    fields = {"Quote": ("q", quote), "Score": ("s", score)}

    train_data, test_data = TabularDataset.splits(
        path="", train="data.json", test="data.json", format="json", fields=fields
    )

    quote.build_vocab(train_data, max_size=10000, min_freq=10, vectors="glove.6B.100d")

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data), batch_size=args.batch_size, device=device, sort=False
    )

    for batch_idx, batch in enumerate(train_iterator):
        # Get data to cuda if possible
        data = batch.q.to(device=device)
        targets = batch.s.to(device=device)
        print(data.dtype)
        print(data)
        print(targets)
        break

