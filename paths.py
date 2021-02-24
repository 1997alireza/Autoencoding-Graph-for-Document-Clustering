import os

root = os.path.dirname(__file__).replace('\\', '/')
if root[-1] != '/':
    root = root + '/'

# directories
src = root + 'src/'
reuters_original_dataset = root + 'venv/share/nltk_data/corpora/reuters/'
the20news_original_dataset = root + 'venv/share/20news-18828/'

# files
reuters_dataset = root + 'datasets/reuters-21578.csv'
the20news_dataset = root + 'datasets/the20news.csv'
