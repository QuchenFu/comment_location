import pandas as pd

data_path = '/tmp/pycharm_project_62/split30_word_data'


def preproces(path, mode):
    df = pd.read_csv(path + "/" + mode + "/data.txt", sep='\t')
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
    data.to_json(path + "/" + mode + "/" + mode + ".json", orient="records", lines=True)


preproces(data_path, "train")
preproces(data_path, "valid")
preproces(data_path, "test")
print("done")
#
# tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
#
# def tokenize(text):
#     tokens, _ = tokenizer.tokenize(text)
#     return tokens
