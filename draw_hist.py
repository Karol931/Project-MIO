import matplotlib.pyplot as plt
import numpy as np



def draw_hist(err):
    counts, bins = np.histogram(err,range=[0,max(err)])
    print(max(err))
    print(min(err))
    plt.hist(bins[:-1], bins, weights=counts)
    plt.yscale('log')
    plt.show()