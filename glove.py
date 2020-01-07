import pickle, string, json, os

from psycopg2.extras import execute_values
from tqdm import trange, tqdm
import psycopg2 as psql
import numpy as np


METADATA = "./glove.json"


class Glove:
    """ Class for pretrained GLOVE vectors
    """


    def __init__(self, metapath, cflag=True):
        """ Takes in metadata file listing database credentials and creates
            Glove object
        """

        with open(metapath, 'r') as infile: meta = json.load(infile)
        self.meta = meta
        self.edict = {}
        
        if self.meta["populated"] == False:

            self.populate_embeddings()
            self.meta["populated"] = True

            with open(METADATA, 'w') as outfile: 
                json.dump(self.meta, outfile, indent=4)

        self.cflag = False
        if cflag: 
            self.build_cache()
            self.cflag = True 


    def build_cache(self):
        """ Loads embeddings and vocabulary as numpy arrays and dict from terms
            to embeddings and loads them in memory

            Data is pickled to keep from having to load on each run
        """

        if os.path.isfile(self.meta["cachepath"]):
            with open(self.meta["cachepath"], 'rb') as infile: cache = pickle.load(infile)
            self.embeddings = cache['embeddings']
            self.vocab = cache['vocab']
            self.vdict = cache['vdict']

        else: 
            self.vocab = np.asarray(self.get_vocab())
            self.embeddings = np.asarray([self[term] for term in tqdm(self.vocab, desc='Retrieving Embeddings')])
            self.vdict = { self.vocab[idx]: idx for idx in range(self.vocab.shape[0]) }

            cache = { 
                'embeddings': self.embeddings,
                'vocab': self.vocab,
                'vdict': self.vdict
            }

            with open(self.meta["cachepath"], 'wb') as outfile: pickle.dump(cache, outfile)


    def get_conn(self):
        """ Returns Postgresql connection object
        """

        return psql.connect(host=self.meta["host"], user=self.meta["user"], database=self.meta["database"], password=self.meta["password"])


    def insert_embeddings(self, conn, vals):
        """ Inserts list of (key, embedding) tuples into Postgresql database
        """

        query = "INSERT INTO embeddings(key, embedding) VALUES %s;"
        cur = conn.cursor()
        execute_values(cur, query, vals)
        conn.commit()


    def process_line(self, line):
        """ Turns line from glove dataset text file into key and numpy array
            embedding
        """

        line = line.strip().split(" ")
        key, embedding = line[0], np.asarray(line[1:])

        return key, embedding


    def populate_embeddings(self):
        """ Populates the glove database with words and their corresponding
            embeddings
        """

        conn, count, seen, vals = self.get_conn(), 0, set(), []

        with open(self.meta["path"], mode="r") as r:
            progress, line = tqdm(total = self.meta["vocab_size"], desc="Populating Embeddings Table"), r.readline()

            while line:
                key, embedding = self.process_line(line)

                if key not in seen: 
                    vals.append((key, json.dumps(embedding.tolist())))

                seen.add(key)

                if count % 1000 == 0:
                    
                    self.insert_embeddings(conn, vals)
                    vals = []

                line = r.readline()
                progress.update(1)
                count += 1


        self.insert_embeddings(conn, vals)


    def get_vocab(self, limit=None):
        """ Returns a list of all the words in the vocabulary
        """

        count, seen, vocab = 0, set(), []
        with open(self.meta["path"], mode="r") as r:
            size = self.meta["vocab_size"] if limit is None else limit
            progress = tqdm(total = size, desc="Retrieving Vocabulary")
            line = r.readline()

            while line:
                
                if limit is not None and count >= limit:
                    break

                term = line.strip().split(" ")[0]
                if term not in seen:
                    vocab.append(term)
                    
                line = r.readline()
                progress.update(1)
                count += 1

        return vocab


    def get_embedding(self, word):
        """ Returns embedding values retrieved from Postgresql database for
            given word
        """

        conn = self.get_conn()
        query = "SELECT embedding FROM embeddings WHERE key=%s LIMIT 1;"
        cur = conn.cursor()
        cur.execute(query, [word])

        return cur.fetchall()


    def __getitem__(self, word):
        """ Returns embedding for given word as a numpy array
            Throws an exception of the word is not in the corpus
        """

        if word in self.edict:
            embedding = self.edict[word]

        else:

            embedding = self.get_embedding(word)

            if len(embedding) > 0:
                embedding = np.asarray(embedding[0][0]).astype(np.float64)
                self.edict[word] = embedding

            else:
                raise KeyError(word)

        return embedding


    def __contains__(self, word):
        """ Determines if the corpus contains the given word
        """

        contains = self.get_embedding(word)

        if len(contains) > 0: contains = True
        else: contains = False

        return contains


class ClusterCache:
    """ Class for clusters of pretrained GLOVE vectors
    """


    def __init__(self, model, k):
        """ Takes in a Glove class instantiation and a cluster size
        """

        self.embeddings_norm = np.linalg.norm(model.embeddings, axis=1)
        self.cembedding_cache = {}
        self.cluster_cache = {}
        self.model = model
        self.k = k


    def get_cluster(self, term):
        """ Takes in a string and returns the cluster of terms nearest to the
            given term and the cluster of embeddings nearest to the embedding of
            the given term
        """

        if term not in self.cluster_cache: 

            embedding = self.model[term]

            num = np.dot(embedding, self.model.embeddings.T)
            den = np.linalg.norm(embedding) * self.embeddings_norm

            csims = ( 1 + (num / den) ) / 2
            cargs = np.argsort(csims)[::-1][:self.k]

            cex = self.model.embeddings[cargs]
            cx = self.model.vocab[cargs]
            
            self.cembedding_cache[term] = cex
            self.cluster_cache[term] = cx

        else:

            cex = self.cembeddings_cache[term]
            cx = self.cluster_cache[term]

        return cex, cx


def main():

    with open(METADATA, mode="r") as r: meta = json.load(r)
    glove = Glove(meta)


if __name__ == "__main__":
    main()
