from collections import defaultdict
from argparse import ArgumentParser


def get_dataset(filename):
    documents = defaultdict(list)
    with open(filename) as freader:
        docs = int(next(freader).strip())
        vocab_size = int(next(freader).strip())
        terms = int(next(freader).strip())
        for line in freader:
            d, v, c = line.strip().split()
            documents[d].append((v, c))
    return documents


def convert_format(documents, filename):
    with open(filename, 'w') as fwriter:
        for document in documents:
            line = '%s %s\n'
            words = ' '.join(['%s:%s' % (key, value)
                              for (key, value) in documents[document]])
            fwriter.write(line % (len(documents[document]), words))


parser = ArgumentParser()
parser.add_argument('-d', '--dataset', help='Input dataset', required=True)
parser.add_argument('-o', '--output', help='Output file', required=True)

args = parser.parse_args()

documents = get_dataset(args.dataset)
convert_format(documents, args.output)
