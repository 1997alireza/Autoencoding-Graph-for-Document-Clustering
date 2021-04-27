# Autoencoding-Graph-for-Document-Clustering

An attempt to reproduce the main results of the paper [Autoencoding Keyword Correlation Graph for Document Clustering](https://www.aclweb.org/anthology/2020.acl-main.366/).

To evaluate and produce the results, you can run [`the main file`](src/main.py).
The results of evaluations are provided in [`this jupyter notebook`](evaluation.ipynb).

<p align="center">
<img src="docs/images/autoencoding-KCG.png?raw=True" alt="Actors Network" width="60%"/>
</p>

Key pieces of the implementation:

### Data Preprocessing
Preprocessing of the datasets [`Reuters-21578`](src/preparation/reuters_cleaning.py) and [`20Newsgroups`](src/preparation/the20news_cleaning.py).

### Keyword Correlation Graph (KCG)
First, the top keywords of documents are extracted to be used as the nodes of KCG ([`Top Keywords Extracting using NMF`](src/modelling/NMF_keyword_extraction.py)).

Then KCG is created ([`KCG Construction`](src/processing/KCG.py)).

### Grarph Autoencoder (GAE)
[`Local Neighborhood Graph Autoencoder (LoNGAE)`](src/modelling/LoNGAE/models/ae.py) is used in this step.

This autoencoder model is applied to KCG ([`Applying the GAE to KCG`](src/processing/GAE_to_KCG.py)).

### Clustering
The clustering methods are applied on the latent space of GAE ([`Clustering`](src/processing/embedding_clustering.py)).

One of the implemented clustering methods in this project is [`Deep Clustering`](src/modelling/deep_clustering/clustering_model.py).
