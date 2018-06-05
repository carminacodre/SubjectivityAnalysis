from trainEN import load_embedding, load_data, preprocess_dataset, preprocess_labels, create_model, crop_and_pad_dataset
from settings import EMBEDDINGS_FILE_RO, TEST_FILE_RO
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

WEGIHTS_FILE = "Checkpoints/subjectivity-ro.11.hdf5"

if __name__ == "__main__":

    #laod the data
    labels, sentences = load_data(TEST_FILE_RO)
    labels = preprocess_labels(labels)

    #load embeddings
    vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_RO, binary=False)
    padding = np.array([np.zeros(300,np.float32)])
    vecs = np.concatenate((vecs,padding), axis=0)

    #load model
    model = create_model(len(vecs), vecs)
    model.load_weights(WEGIHTS_FILE)

    sentences = preprocess_dataset(vocabulary, sentences)
    sentences = crop_and_pad_dataset(vocabulary, sentences)

    #convert to np arrays
    sentences = np.array(sentences)
    labels = np.array(labels)

    predictions = model.predict(sentences)
    predictions = [1 if i > 0.5 else 0 for i in predictions]

    print(classification_report(labels, predictions))
    print(accuracy_score(labels, predictions))
    print(confusion_matrix(labels, predictions))
