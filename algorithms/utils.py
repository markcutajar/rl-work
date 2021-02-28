"""
This file is not entirely my content, i've been collecting stuff from different creators.

1. Function 'plot_learning_curve' is based on philtabor's work: http://bit.ly/37P8MKY
"""






import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 8))


def plot_learning_curve(x, scores, figure_file, name=None):

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    line, = plt.plot(x, running_avg)
    line.set_label(name)
    plt.title('Running average of previous 100 scores')
    plt.legend()
    plt.savefig(figure_file)