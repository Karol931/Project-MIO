import sys

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import fasttext

if __name__ == "__main__":
    fasttext_model_name = "model_fasttext_3.bin"
    if len(sys.argv) > 1:
        fasttext_model_name = sys.argv[1]

    if os.path.exists(fasttext_model_name):
        model = fasttext.load_model(fasttext_model_name)
        test = "make america great again"
        if len(sys.argv) > 2:
            test = sys.argv[2]
        print(model.predict(test, k=2))
        # print(model.get_nearest_neighbors(test))