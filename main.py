import sys
from nlp import create_model
from model_classification import show_fasttext_model

if __name__ == "__main__":
        if len(sys.argv) > 1:
                if sys.argv[1] == "regression":
                        if len(sys.argv) > 2:
                                if sys.argv[2] == "log":
                                        create_model(sys.argv[2])
                                elif sys.argv[2] == "quant":
                                        create_model(sys.argv[2])
                        else:
                                pass
                elif sys.argv[1] == "classification":
                        if len(sys.argv) > 2:
                                if sys.argv[2] == "log":
                                        show_fasttext_model(True)
                                elif sys.argv[2] == "quant":
                                        show_fasttext_model(False)
                        else:
                                show_fasttext_model(True)
        else:
                print("provide arguments:")
                print("python ./main.py [regression/classification] [log/quant]")

