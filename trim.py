import argparse, string, re

from nltk.tag import StanfordPOSTagger, UnigramTagger
from nltk.corpus import masc_tagged, treebank, brown
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-l", "--linecount", required=True)
    args = parser.parse_args()
    args.linecount = int(args.linecount)

    # Load stopwords & add/remove as necessary
    stops = stopwords.words("english")
    stops.append('"')

    for term in [ 
        'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
        'it', 'they', 'them', 'their', 'theirs', 'themselves', 'against',
        'between', 'above', 'below', 'up', 'down', 'over', 'under', 'few',
        'more', 'most' 
    ]: stops.remove(term)

    count = 0
    with open(args.output, mode='w') as w:
        with open(args.input, mode='r') as r:
            progress = tqdm(total=args.linecount)
            line = r.readline()

            while line:

                if count >= args.linecount: break

                term = line.strip().split(' ')[0]

                if not any(char in string.punctuation for char in term) and \
                   not any(char.isdigit() for char in term) and \
                   all(char.isalnum() for char in term) and \
                   term not in stops: 

                    w.write(line)
                    count += 1

                line = r.readline()
                progress.update(1)


if __name__ == '__main__':
    main()
