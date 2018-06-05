from trainEN import create_model
from data_utils import *
from settings import EMBEDDINGS_FILE_EN, EMBEDDINGS_FILE_RO
from keras import Model
import numpy as np
import argparse

WEGIHTS_FILE_EN = "Checkpoints/subjectivity-en.15.hdf5"
WEGIHTS_FILE_RO = "Checkpoints/subjectivity-ro.11.hdf5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict if sentence objective or subjective')
    parser.add_argument('-l', dest='language', help='language EN/RO')
    parser.add_argument('-s', dest='input', help='sentence')
    args = parser.parse_args()
    language= args.language
    input = args.input

    if language == 'EN':
        # load embeddings
        vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_EN)
        padding = np.array([np.zeros(300, np.float32)])
        vecs = np.concatenate((vecs, padding), axis=0)

        # load model
        model = create_model(len(vecs), vecs)
        model = Model()
        model.load_model(WEGIHTS_FILE_EN)

    else:
        # load embeddings
        vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_RO, binary=False)
        padding = np.array([np.zeros(300, np.float32)])
        vecs = np.concatenate((vecs, padding), axis=0)

        # load model
        model = create_model(len(vecs), vecs)
        model.load_weights(WEGIHTS_FILE_RO)

    sentences = preprocess_dataset(vocabulary, [input])
    sentences = crop_and_pad_dataset(vocabulary, sentences)

    # convert to np arrays
    sentences = np.array(sentences)

    predictions = model.predict(sentences)

    print(predictions[0])
    if predictions[0] > 0.5:
        print("Subjective")
    else:
        print("Objective")
