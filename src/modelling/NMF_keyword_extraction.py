"""
extracting most-related keywords to input sentences based on NMF on TF-IDF matrix of the sentences
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from src.utils.mathematical import cosine_similarity
from collections import defaultdict
import numpy as np


def extract_top_keywords(documents_sentences, max_number=50):
    """
    :param documents_sentences: a 2d list of sentences. the sentences of each row are related to one document
    :param max_number: maximum number of returned keywords
    :return: top keywords with the list of related sentences: dictionary of {keyword: [document index, sentence index in document]}
    """

    flatten_sentences = []
    indexes_mapping = {}  # indexes_mapping[i, j] = k; means documents[i, j] = flatten_sentences[k]
    for i, doc in enumerate(documents_sentences):
        for j, sent in enumerate(doc):
            indexes_mapping[i, j] = len(flatten_sentences)
            flatten_sentences.append(sent)

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tfidf = vectorizer.fit_transform(flatten_sentences)
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

    keyword_sents = defaultdict(list)
    # dictionary of {keyword: [(i,j)] list of index of related sentences to the keyword};
    # i is the index of document, and j is the index of sentence in the ith document

    dummy_sentences = []
    # [(i,j)]; a list of sentences that doesn't contain any of top keywords. they will attached to a dummy node (keyword)

    for i, doc in enumerate(documents_sentences):
        for j, sent in enumerate(doc):
            flatten_idx = indexes_mapping[i, j]
            sent_tfidf = tfidf[flatten_idx].toarray()[0]
            similarities_to_topic = [cosine_similarity(sent_tfidf, topic) for topic in nmf.components_]
            matched_topic_id = np.array(similarities_to_topic).argmax()
            matched_topic = nmf.components_[matched_topic_id]
            matched_key, match_score = None, .0
            for key_id in top_keys:  # find most-related keyword to the sentence among the top keywords
                if matched_topic[key_id] * sent_tfidf[key_id] > match_score:  # multiplication of tf-idf of the keyword in topic and sentence
                    match_score = matched_topic[key_id] * sent_tfidf[key_id]
                    matched_key = key_id
                    # TODO: maybe is different from the original paper

            if matched_key is not None:
                # it's possible that a document doesn't contain any of the top keywords
                matched_keyword = feature_names[matched_key]
                keyword_sents[matched_keyword].append((i, j))
            else:  # it's a dummy sentence
                dummy_sentences.append((i, j))

    keyword_sents['THE_DUMMY_NODE'] = dummy_sentences
    return keyword_sents
