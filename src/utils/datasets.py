import paths
import csv
import numpy as np


def fetch_dataset(dataset_path=paths.the20news_dataset):
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
    return np.array(data, dtype='object')
