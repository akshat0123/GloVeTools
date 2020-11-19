import pickle, json, os, re

from tqdm import tqdm
import numpy as np


def load_glove_embeddings_from_file(glovepath, limit=float('inf')):
    """ Loads numpy array of glove vocabulary and embeddings

        Args:
            glovepath (string): path to GloVe pretrained embeddings file
            limit (int): number of vocabulary terms to return embeddings for

        Returns:
            (numpy.ndarray): array of all vocabulary terms
            (numpy.ndarray): array of all embeddings
    """

    # Set up progress bar
    linecount = sum([1 for line in open(glovepath, 'r')])
    progress = tqdm(total=linecount)
    progress.update(1)

    # Read glove embeddings file
    count = 1
    with open(glovepath, 'r') as infile:
        terms, embeddings = [], []

        line = infile.readline()
        while line:
            line = line.split(' ')
            terms.append(line[0])
            embeddings.append(line[1:])

            count += 1
            if count > limit: break

            line = infile.readline()
            progress.update(1)

    return np.asarray(terms), np.asarray(embeddings).astype(np.float64)


def load_glove_embeddings_from_cache(cachepath):
    """ Loads numpy array of glove vocabulary and embeddings from pickled cache

        Args:
            cachepath (string): path to cache for pretrained embeddings

        Returns:
            (numpy.ndarray): array of all vocabulary terms
            (numpy.ndarray): array of all embeddings
    """

    cache = pickle.load(open(cachepath, 'rb'))
    terms, embeddings = cache['terms'], cache['embeddings']
    return terms, embeddings


def load_glove_embeddings(glovepath, cachepath, limit=float('inf')):
    """ Loads numpy array of glove vocabulary and embeddings

        Args:
            glovepath (string): path to GloVe pretrained embeddings file
            cachepath (string): path where cached result is to be saved
            limit (int): number of vocabulary terms to return embeddings for

        Returns:
            (numpy.ndarray): array of all vocabulary terms
            (numpy.ndarray): array of all embeddings
    """

    if os.path.isfile(cachepath):
        terms, embeddings = load_glove_embeddings_from_cache(cachepath)

        if len(terms) != limit:
            terms, embeddings = load_glove_embeddings_from_file(glovepath, limit=limit)
            cache = { 'terms': terms, 'embeddings': embeddings }
            pickle.dump(cache, open(cachepath, 'wb'))

    else:
        terms, embeddings = load_glove_embeddings_from_file(glovepath, limit=limit)
        cache = { 'terms': terms, 'embeddings': embeddings }
        pickle.dump(cache, open(cachepath, 'wb'))

    return terms, embeddings


def tokenize(string, glove):
    """ Tokenize string into list of unique in-vocabulary words

        Args:
            string (string): string to be tokenized
            glove (NLP4FinTools.glove.Glove): Glove language model class

        Returns:
            (list): list of tokens
    """

    string = re.sub("[^a-zA-Z]", " ", string)
    tokens = [token for token in string.split(' ') if token != '']
    tokens = [token for token in tokens if token not in glove.stops]
    tokens = [token for token in tokens if token in glove]
    tokens = list(set(tokens))

    return tokens


