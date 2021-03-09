"""
extracting most-related keywords to input sentences based on NMF on TF-IDF matrix of the sentences
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from src.utils.mathematical import cosine_similarity
from collections import defaultdict
import numpy as np


def extract_top_keywords(sentences, max_number=50):
    """

    :param sentences:
    :param max_number: maximum number of returned keywords
    :return: top keywords with the list of related sentences: dictionary of {keyword: list of related sentences}
    """

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names()

    nmf = NMF().fit(tfidf)
    keys_score = []
    for topic_idx, topic in enumerate(nmf.components_):
        for keyword_idx, keyword_score in enumerate(topic):
            if keyword_score > .0:
                keys_score.append((keyword_idx, keyword_score))

    keys_score.sort(key=lambda item: item[1], reverse=True)
    top_keys = []
    for item in keys_score:
        if item[0] not in top_keys:
            top_keys.append(item[0])
        if len(top_keys) == max_number:
            break

    keyword_sents = defaultdict(list)  # dictionary of {keyword: list of related sentences to the keyword}

    for idx in range(len(sentences)):
        sent_tfidf = tfidf[idx].toarray()[0]
        similarities_to_topic = [cosine_similarity(sent_tfidf, topic) for topic in nmf.components_]
        matched_topic_id = np.array(similarities_to_topic).argmax()
        matched_topic = nmf.components_[matched_topic_id]
        matched_key, match_score = None, -1
        for key_id in top_keys:  # find most-related keyword to the sentence among the top keywords
            if matched_topic[key_id] * sent_tfidf[key_id] > match_score:  # multiplication of tf-idf of the keyword in topic and sentence
                match_score = matched_topic[key_id] * sent_tfidf[key_id]
                matched_key = key_id

        matched_keyword = feature_names[matched_key]
        keyword_sents[matched_keyword].append(idx)

    return keyword_sents
