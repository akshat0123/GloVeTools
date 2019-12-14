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
    parser.add_argument("-sp", "--spos", action="store_true")
    parser.add_argument("-p", "--pos", action="store_true")
    args = parser.parse_args()
    args.linecount = int(args.linecount)

    # Load stopwords & add/remove as necessary
    stops = stopwords.words("english")
    stops.append('"')
    for term in [ 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 'they', 'them', 'their', 'theirs', 'themselves', 'against', 'between', 'above', 'below', 'up', 'down', 'over', 'under', 'few', 'more', 'most' ]:
        stops.remove(term)

    # Parts of speech that will be kept
    keep_pos = [ "NN", "JJ" ]

    stanford_tagger_model = './stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
    stanford_tagger_jar = './stanford-postagger-2018-10-16/stanford-postagger.jar'
    stanford_tagger = StanfordPOSTagger(stanford_tagger_model, stanford_tagger_jar)

    t1 = UnigramTagger(masc_tagged.tagged_sents())
    t2 = UnigramTagger(brown.tagged_sents())
    t3 = UnigramTagger(treebank.tagged_sents())
    unigram_taggers = [t1, t2, t3]

    count = 0
    with open(args.output, mode='w') as w:
        with open(args.input, mode='r') as r:
            progress = tqdm(total=args.linecount)
            line = r.readline()

            while line:

                if count >= args.linecount: break

                term = line.strip().split(' ')[0]

                if not any(char in string.punctuation for char in term) and \
                   not any(not char.isalnum() for char in term) and \
                   not any(char.isdigit() for char in term) and \
                   term not in stops:

                    if args.spos:
                        tag = stanford_tagger.tag([term])[0][1]
                        if tag in keep_pos:
                            w.write(line)
                            count += 1

                    elif args.pos:
                        tags = [tagger.tag([term])[0][1] for tagger in unigram_taggers]
                        if any(tag in keep_pos for tag in tags):
                            w.write(line)
                            count += 1

                    else:
                        w.write(line)
                        count += 1

                line = r.readline()
                progress.update(1)


if __name__ == '__main__':
    main()
