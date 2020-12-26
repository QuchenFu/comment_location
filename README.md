# comment_location

This git repo built a pytorch data loader for the comment location prediction dataset. The train.py also contains a vanilla LSTM model that can achieve 0.76 accuracy and 0.26 recall which is better than the original paper.

How to run(wait for the `glove.6B.100d` to be download in line4):

`git clone https://github.com/QuchenFu/comment_location.git`

`cd comment_location`

`pip3 install -r requirements.txt`

`python3 show_batch.py --batch_size=2 --file_path="split30_word_data/train/data.txt"`

`python3 show_batch.py --batch_size=8 --file_path="split30_word_data/train/data.txt"`

Note:
The show_batch.py file is only for demonstration. It uses the train data also as the test(wrong) since only one parameter of the location of data.txt is allowed. I printed out the tensor type, value of the source, and value of the target. The tokenizer is from OpenNMS and may not be the best case, also the embedding is glove.6B.100d which may not be suitable.
