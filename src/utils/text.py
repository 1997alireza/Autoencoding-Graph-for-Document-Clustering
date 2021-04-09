from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


def split_document(doc):
    sentences = doc.split('.')
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(len, sentences))
    return sentences


def preprocess(text):
    """pre-processing documents before using them for creating the KCG"""
    text = text.lower()
    text_p = "".join([char for char in text if char not in
                      '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~1234567890'])  # removing punctuations except dot (.), and numbers
    words = word_tokenize(text_p)
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    stemmed_text = ''
    for w in words:
        if w not in stop_words:
            stemmed_text += porter.stem(w) + ' '

    stemmed_text = re.sub(r"[ .]*[.][ ]*", " . ", stemmed_text)
    return stemmed_text[:-1]
