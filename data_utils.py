import codecs
from gensim.models import KeyedVectors
from sklearn.utils import shuffle

DEFAULT_SEQUENCE_LENGTH = 30

def load_data(file_path):
    print("Loading data %s ..." %file_path)
    with codecs.open(file_path, "r",encoding='utf-8', errors='ignore') as data_file:
        samples = data_file.read()
        samples = samples.split("\n")

        # remove <context ...> lines
        clean_samples = []
        for s in samples:
            if not s.startswith('<context'):
                clean_samples.append(s)
        del clean_samples[len(clean_samples) - 1]

        samples = clean_samples
        annotations = [i.split(" ")[0] for i in samples]
        sentences = [i[2:] for i in samples]

        assert(len(annotations) == len(sentences))
        print("Number of samples %d" % len(annotations))

        return annotations, sentences

def load_data_sen(file_path):
    print("Loading data %s ..." %file_path)
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as data_file:
        samples = data_file.read()
        samples = samples.split("\n")
        print("Number of samples %d" % len(samples))
        return samples

def load_embedding(embedding_path, binary=True):
    print("Loading word embedding %s ..." %embedding_path)
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=binary)
    vectors = word_vectors.vectors
    vocabulary = word_vectors.vocab

    return vocabulary, vectors

def preprocess_sentence(vocabulary, sentence):
    s = sentence.strip().lower()
    s = s.split(" ")
    tokens = [vocabulary[word].index for word in s if word in vocabulary]
    return tokens

def preprocess_dataset(vocab, sentences):
    result = []
    for s in sentences:
        result.append(preprocess_sentence(vocab, s))
    return result

def preprocess_labels(labels):
    result = []
    for l in labels:
        if l == 'O':
            result.append(0)
        else:
            result.append(1)
    return result

def crop_and_pad_dataset(vocab, sentences, sequence_length=DEFAULT_SEQUENCE_LENGTH):
    result = []
    pad_index = len(vocab)
    for s in sentences:
        if len(s) > sequence_length:
            result.append(s[:sequence_length])
        else:
            result.append(s + [pad_index] * (sequence_length - len(s)))
    return result

def shuffle_data(sentences, labels):
    return shuffle(sentences, labels)

def extract_save_test(sentences, labels, p, file_path):
    num_samples = len(sentences)
    test_sentences = sentences[int(num_samples-p*num_samples):]
    train_sentences = sentences[:int(num_samples-p*num_samples)]
    test_labels = labels[int(num_samples-p*num_samples):]
    train_labels = labels[:int(num_samples-p*num_samples)]

    #save to file
    with open(file_path, "w") as f:
        for (s, l) in zip(test_sentences, test_labels):
            if l==0:
                ls='O'
            else:
                ls='S'
            f.write("%s %s\n" %(ls, s))
    print("Test data written to %s ..." %file_path)
    return train_sentences, train_labels

