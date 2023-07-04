import sys
from nlp import create_model

if __name__ == "__main__":

        if sys.argv[1] == "log":
                create_model(sys.argv[1])
        elif sys.argv[1] == "quant":
                create_model(sys.argv[1])
        elif sys.argv[1] == "fasttext":
                pass


