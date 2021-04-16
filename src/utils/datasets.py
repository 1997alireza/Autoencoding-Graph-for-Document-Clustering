import csv
import numpy as np


def fetch_dataset(dataset_path):
    """

    :param dataset_path: paths.reuters_dataset or paths.the20news_dataset
    :return: numpy array of [label, document]
    """

    with open(dataset_path) as ds:
        reader = csv.reader(ds, delimiter=',')
        headers = next(reader)
        data = []
        for row in reader:
            data.append(row)
    print('dataset is loaded')
    return np.array(data, dtype='object')


def name_of_dataset(dataset_path):
    return dataset_path.split('/')[-1].split('.')[0]


def number_of_labels(dataset_name):
    if dataset_name == 'reuters-21578':
        return 10
    elif dataset_name == 'the20news':
        return 20

    raise Exception('dataset is not found')
