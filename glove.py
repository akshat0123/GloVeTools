from NLP4FinTools.glove.utils import load_glove_embeddings
from nltk.corpus import stopwords


class Glove:


    def __init__(self, glovepath, cachepath, limit=100000):
        """
        """

        # Load embeddings 
        self.terms, self.embeds = load_glove_embeddings(glovepath, cachepath, limit=limit)
        self.tdict = { self.terms[idx]: idx for idx in range(self.terms.shape[0]) }

        # Load stopwords and add capitalized versions of stopwords
        stops = stopwords.words('english')
        self.stops = set(stops + [stop[0].upper() + stop[1:] for stop in stops])


    def __contains__(self, term):
        """ Check if term is in word embedding set

            Args:
                term (string): Term to check
            
            Returns:
                (bool): Whether term exists in vocabulary
        """

        contains = True if term in self.tdict else False
        return contains


    def __getitem__(self, term):
        """ Return embedding for term if its in the vocabulary

            Args:
                term (string): Term embedding to return

            Return:
                (numpy.ndarray): Numpy array of word embedding
        """

        return self.embeds[self.tdict[term]]


