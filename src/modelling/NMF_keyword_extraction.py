"""
extracting most-related keywords to input sentences based on NMF on TF-IDF matrix of the sentences
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from src.utils.mathematical import cosine_similarity
from collections import defaultdict
import numpy as np
import pickle
import paths

THE_DUMMY_NODE = 'THE_DUMMY_NODE'


def extract_top_keywords(documents_sentences, max_number=50, dataset_name=None):
    """
    :param documents_sentences: a 2d list of sentences. the sentences of each row are related to one document
    :param max_number: maximum number of returned keywords
    :param dataset_name: is used to save or load top keywords
    :return: top keywords with the list of related sentences: dictionary of {keyword: [document index, sentence index in document]}
    """

    keywords_file_path = None
    if dataset_name is not None:
        keywords_file_path = paths.models + 'document_keywords_set/' + dataset_name + '.pkl'
        try:
            keyword_sents = pickle.load(open(keywords_file_path, 'rb'))
            print('top keywords of documents are loaded')
            return keyword_sents
        except FileNotFoundError:
            pass

    indexes_mapping = {}  # indexes_mapping[i, j] = k; means documents[i, j] = flatten_sentences[k]

    def sentences_iterator():
        index = 0
        for i, doc in enumerate(documents_sentences):
            for j, sent in enumerate(doc):
                indexes_mapping[i, j] = index
                index += 1
                yield sent

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=300, stop_words='english')
    tfidf = vectorizer.fit_transform(sentences_iterator())

    feature_names = vectorizer.get_feature_names()

    print('Applying NMF decomposition...\n')
    nmf = NMF(verbose=True).fit(tfidf)
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
        print('doc {}/{}'.format(i, len(documents_sentences)))
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

            if matched_key is not None:
                # it's possible that a document doesn't contain any common keywords with the matched topic
                matched_keyword = feature_names[matched_key]
                keyword_sents[matched_keyword].append((i, j))
            else:  # it's a dummy sentence
                dummy_sentences.append((i, j))

    keyword_sents[THE_DUMMY_NODE] = dummy_sentences
    # for reuters-21578 dataset, 84.61% sentences are assigned to the dummy node!
    # for the20news dataset, 86.93% (269509/310018) sentences are assigned to the dummy node!

    if keywords_file_path is not None:
        pickle.dump(keyword_sents, open(keywords_file_path, 'wb'))
    print('top keywords are extracted')
    return keyword_sents
