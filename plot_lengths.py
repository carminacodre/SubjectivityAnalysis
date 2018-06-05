from trainRO import load_embedding, load_data, preprocess_dataset
from trainEN import load_data_sen
from settings import DATA_FILE_RO, EMBEDDINGS_FILE_RO,  OBJ_FILE_EN, SUBJ_FILE_EN, EMBEDDINGS_FILE_EN
import matplotlib.pyplot as plt
from collections import Counter

ENGLISH = 0


def test_above(limit, sentences):
    above = 0
    for split_t in sentences:
        if len(split_t) > limit:
            above+=1
    return above*1.0/len(sentences)*100

if __name__ == "__main__":

    #laod the data

    if ENGLISH:
        sentences_obj = load_data_sen(OBJ_FILE_EN)
        labels_obj = [0] * len(sentences_obj)

        sentences_subj = load_data_sen(SUBJ_FILE_EN)
        labels_subj = [1] * len(sentences_subj)

        sentences= sentences_obj + sentences_subj
        labels = labels_obj + labels_subj
    else:
        labels, sentences = load_data(DATA_FILE_RO)

    annotations_stats = Counter(labels)
    print(annotations_stats.keys())
    print(annotations_stats.values())

    #load embeddings
    if ENGLISH:
        vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_EN)
    else:
        vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_RO, binary=False)

    sentences = preprocess_dataset(vocabulary, sentences)

    lengths = []
    for s in sentences:
        lengths = lengths + [len(s)]

    print("%f sentences are above 30 words" % test_above(30, sentences))

    numpy_hist = plt.figure(figsize=(25, 15))
    plt.hist(lengths, bins=range(0, 600))
    plt.show()