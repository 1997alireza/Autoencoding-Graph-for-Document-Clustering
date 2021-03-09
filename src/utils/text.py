def split_document(doc):
    sentences = doc.split('.')
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(len, sentences))
    return sentences
