# comment_location

This git repo built a pytorch data loader for the comment location prediction dataset. The train.py also contains a vanilla LSTM model that can achieve 0.76 accuracy and 0.26 recall which is better than the original paper.

How to run:

`git clone https://github.com/QuchenFu/comment_location.git`

`cd comment_location`

`sh show_batch.sh 1 split30_word_data/train/data.txt`

`sh show_batch.sh 8 split30_word_data/train/data.txt`

The first parameter is batch size, the second parameter is the data file path


Or step by step(wait for the `glove.6B.100d` to be downloaded in line 4):

`git clone https://github.com/QuchenFu/comment_location.git`

`cd comment_location`

`pip3 install -r requirements.txt`

`python3 show_batch.py --batch_size=2 --file_path="split30_word_data/train/data.txt"`

`python3 show_batch.py --batch_size=8 --file_path="split30_word_data/train/data.txt"`

Note:
The show_batch.py file is only for demonstration. Train and test are the same since only one parameter of the location of data.txt is allowed. I printed out the tensor type, value of the source, and value of the target. The tokenizer is from OpenNMT and may not be the best case, also the embedding is glove.6B.100d which may not be suitable. 

Still, the model performs well with spacy_en tokenizer + glove.6B.100d on vanilla LSTM with simple LOCs concatenation within each snnipet(code in train.py).
