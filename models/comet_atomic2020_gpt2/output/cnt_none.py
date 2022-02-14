import argparse
import json
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--k", type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    data = [ json.loads(l) for l in open(args.input_file, 'r').readlines()]
    nones = [1 if all(np.array(tu['generation'][:args.k])=='none') else 0  for tu in data]
    print(f"k: {args.k}, length of dataset: {len(data)}, none count: {sum(nones)}, portion: {sum(nones) / len(data)}")