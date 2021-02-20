"""
Data cleansing based on the details in the paper
"Determining Gains Acquired from Word Embedding Quantitatively Using Discrete Distribution Clustering"
"""

from nltk.corpus import reuters
from collections import defaultdict
import csv
import paths


def prepare_reuters():
    id2cat = defaultdict(list)

    for line in open(paths.reuters_original_dataset + '/cats.txt', 'r'):
        doc_id, _, cats = line.partition(' ')
        cats = cats.split()
        if len(cats) == 1:  # only add documents with single category
            id2cat[doc_id] = cats[0]

    # remove duplicates
    docids = list(id2cat.keys())
    to_remove_docids = []
    for idx, doc_id0 in enumerate(docids):
        doc0 = reuters.raw(doc_id0)
        for doc_id1 in docids[idx+1:]:
            doc1 = reuters.raw(doc_id1)
            if doc0 == doc1:
                to_remove_docids.append(doc_id0)
                to_remove_docids.append(doc_id1)

    for doc_id in to_remove_docids:
        if doc_id in id2cat:
            del id2cat[doc_id]

    cat2docids = defaultdict(list)
    for doc_id in list(id2cat.keys()):
        cat = id2cat[doc_id]
        cat2docids[cat].append(doc_id)

    # select top ten largest categories
    top_cat2docids = dict(sorted(cat2docids.items(), key=lambda item: len(item[1]), reverse=True)[:10])

    with open(paths.reuters_dataset, mode='w') as ds:  # saving cleaned database
        ds_writer = csv.writer(ds, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ds_writer.writerow(['category', 'document'])  # column names

        for cat in top_cat2docids:
            for doc_id in top_cat2docids[cat]:
                doc = reuters.raw(doc_id)
                ds_writer.writerow([cat, doc])


def read():  # 7884 documents in 10 categories
    with open(paths.reuters_dataset) as ds:
        reader = csv.reader(ds, delimiter=',')
        headers = next(reader)
        for row in reader:
            print('Category:  ', row[0])
            print('Document:  ', row[1])
