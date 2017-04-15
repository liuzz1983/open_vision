import numpy as np
import sys
import argparse


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('embeddings', type=str, nargs='+',
        help='Model definition. Points to a module containing the definition of the inference graph.')

    return parser.parse_args(argv)

def main():
	args = parse_arguments(sys.argv[1:])

	group_image = args.embeddings[0]
	person_image = args.embeddings[1]

	group_embedding = np.load(group_image)
	person_embedding = np.load(person_image)

	for i, p in enumerate(person_embedding):
		diff = group_embedding - p
		dist = np.sum( np.square(diff), axis=1)

		index = np.argmin(dist)

		print i, "----", index, ":", dist[index]

if __name__ == "__main__":
	main()