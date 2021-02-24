"""
extracting most-related keyword to each document based on TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_top_keywords(documents):
    """

    :param documents:
    :return: extracted keyword of each document, and transformer function
    """
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    tfidfs = vectorizer.fit_transform(documents)  # fit, then transform
    vocab = vectorizer.get_feature_names()
    return [vocab[tfidfs[i].argmax()] for i in range(len(documents))], vectorizer.transform
