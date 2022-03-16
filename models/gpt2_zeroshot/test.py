import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int)
parser.add_argument("--lst", nargs="+")
parser.add_argument("--b", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)