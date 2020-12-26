git clone https://github.com/QuchenFu/comment_location.git

cd comment_location

pip3 install -r requirements.txt

python3 show_batch.py --batch_size=2 --file_path="split30_word_data/train/data.txt"

python3 show_batch.py --batch_size=8 --file_path="split30_word_data/train/data.txt"