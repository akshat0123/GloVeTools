import string, time, json

from psycopg2.extras import execute_values
from tqdm import trange, tqdm
import psycopg2 as psql
import numpy as np


METADATA = "./glove.json"


def cosine_similarity(a, b):
    """ Returns cosine similarity of vectors 'a' and 'b'
    """

    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)

    return (1 + (num/den)) / 2


class Glove:
    """ Class for pretrained GLOVE vectors
    """


    def __init__(self, meta):
        """ Takes in metadata object listing database credentials and creates
            Glove object
        """

        self.meta = meta
        self.cache = {}
        
        if self.meta["populated"] == False:
            self.populate_embeddings()
            self.populate_distances()


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


    def populate_distances(self):
        """ Populates the glove database with pairwise cosine similarities of
            all the embeddings in the vocabulary
        """

        embeddings = self.get_embedding_dict()
        vocab = list(embeddings.keys())
        punctuation = string.punctuation

        vals = []

        with open(self.meta["distances_path"], mode="w") as w:
            for aidx in trange(len(vocab)):
                for bidx in range(1, len(vocab)-aidx):
                    
                    term_a, term_b = vocab[aidx], vocab[aidx + bidx]
                    embed_a, embed_b = embeddings[term_a], embeddings[term_b]
                    distance = cosine_similarity(embed_a, embed_b)

                    if term_a in punctuation: term_a = "\\" + term_a
                    if term_b in punctuation: term_b = "\\" + term_b

                    w.write('"%s"|"%s"|%.6f\n' % (term_a, term_b, distance))


    def get_embedding_dict(self, limit=None):
        """ Returns a dictionary of all words in the vocabulary mapped to their
            embeddings
        """

        count, embeddings, seen = 0, {}, set()
        with open(self.meta["path"], mode="r") as r:
            progress = tqdm(total = self.meta["vocab_size"], desc="Retrieving Embeddings")
            line = r.readline()

            while line:

                if limit is not None and count >= limit: 
                    break

                key, embedding = self.process_line(line)

                if key not in seen: 
                    embeddings[key] = embedding.astype(np.float64)

                line = r.readline()
                progress.update(1)
                count += 1

        return embeddings


    def get_cluster_dict(self, limit=None):
        """ Returns a 2D dictionary with the first key specifiying a centroid
            term and the second key specifying the closest term by rank

            e.g.: clusters['animal'][2] => the 3rd closest word to 'animal'
        """

        query = "SELECT * FROM clusters ORDER BY term_a, distance DESC"
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()

        clusters, rank, seen = {}, 1, set()
        for row in rows:

            term_a, term_b, distance = row[0], row[1], row[2]

            if term_a in seen:
                rank += 1

            else:
                clusters[term_a] = { 0: term_a }
                seen.add(term_a)
                rank = 1

            clusters[term_a][rank] = term_b

        return clusters


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


    def get_cluster(self, centroid, k):
        """ Returns 'centroid' and 'k-1' closest words to 'centroid' in the
            embeddings space
        """

        query = "SELECT term_b FROM clusters where term_a=%s order by distance desc limit %s;"
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(query, [centroid, k-1])

        results = cur.fetchall()
        if len(results) > 0:
            results = [term[0] for term in results]
            results.insert(0, centroid)

        return results


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

        if word in self.cache:
            embedding = self.cache[word]

        else:

            embedding = self.get_embedding(word)

            if len(embedding) > 0:
                embedding = np.asarray(embedding[0][0]).astype(np.float64)
                self.cache[word] = embedding

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


def main():

    with open(METADATA, mode="r") as r: meta = json.load(r)
    glove = Glove(meta)


if __name__ == "__main__":
    main()
