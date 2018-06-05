from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from data_utils import *
from settings import DATA_FILE_RO, EMBEDDINGS_FILE_RO, CHECKPOINTS_PATH, TENSORBOARD_PATH, TEST_FILE_RO
import os
import numpy as np

#Parametres
embedding_dim = 300
filter_sizes = [3, 4, 5]
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
sequence_length = 30

#Training parameters
batch_size = 8
num_epochs = 20
val_split = 0.1
lr = 0.0005

test_p=0.15

def create_model(dictionary_dim, vecs):

    #parallel convolutions
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(activation="relu", padding="valid", strides=1, filters=num_filters, kernel_size=fsz)(
            graph_in)
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes) > 1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]

    graph = Model(inputs=graph_in, outputs=out)
    graph.summary()

    model = Sequential()
    model.add(Embedding(input_dim=dictionary_dim, output_dim=embedding_dim, input_length=sequence_length, weights=[vecs],
                        trainable=False))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = SGD(lr=lr, momentum=0.80, decay=1e-6, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

def create_checkpoints():
    cp_cb = ModelCheckpoint(filepath=os.path.join(CHECKPOINTS_PATH, "subjectivity-ro.{epoch:02d}.hdf5"), monitor='val_loss',
                            save_best_only=True)
    tb_cb = TensorBoard(log_dir=TENSORBOARD_PATH)
    return [cp_cb, tb_cb]

if __name__ == "__main__":

    #laod the data
    labels, sentences = load_data(DATA_FILE_RO)
    labels = preprocess_labels(labels)

    #extract and save test data
    sentences, labels = shuffle_data(sentences, labels)
    sentences, labels = extract_save_test(sentences, labels, test_p, TEST_FILE_RO)

    #load embeddings
    vocabulary, vecs = load_embedding(EMBEDDINGS_FILE_RO, binary=False)
    padding = np.array([np.zeros(300,np.float32)])
    vecs = np.concatenate((vecs,padding), axis=0)

    #load model
    model = create_model(len(vecs), vecs)

    sentences = preprocess_dataset(vocabulary, sentences)
    sentences = crop_and_pad_dataset(vocabulary, sentences)

    #convert to np arrays
    sentences = np.array(sentences)

    #checkpoints
    ckpt_list = create_checkpoints()

    model.fit(sentences, labels, batch_size=batch_size,
              epochs=num_epochs, validation_split=val_split, verbose=1, callbacks=ckpt_list)

    print("Done")
