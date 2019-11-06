import argparse, string, re

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
    for term in [ 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 'they', 'them', 'their', 'theirs', 'themselves', 'against', 'between', 'above', 'below', 'up', 'down', 'over', 'under', 'few', 'more', 'most' ]:
        stops.remove(term)

    # Parts of speech that will be trimmed from dataset
    trim_pos = [ "$", "''", "(", ")", ",", "--", ".", ":", "CC", "DT", "EX", "MD", "POS", "SYM", "TO", "WDT", "WP", "WP$", "WRB", "``" ]

    count = 0
    with open(args.output, mode='w') as w:
        with open(args.input, mode='r') as r:
            progress = tqdm(total=args.linecount)
            line = r.readline()

            while line:

                if count >= args.linecount: break

                term = line.strip().split(' ')[0]
                tag = nltk.pos_tag([term])[0][1]

                if tag not in trim_pos and term not in stops and term not in string.punctuation and re.match('[A-Za-z0-9]', term) and not term.isdigit():
                    w.write(line)
                    count += 1

                line = r.readline()
                progress.update(1)


if __name__ == '__main__':
    main()
