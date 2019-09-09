import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def import_file(path):
    results = []
    i = 0
    with open(path, 'r') as infile:
        for line in infile.readlines():
            linedata = {}
            for match in re.findall("([a-z_]*?): (\d+\.\d+)", line):
                linedata[match[0]] = float(match[1])
            #linedata['index'] = i
            if 'ccuracy' not in linedata.keys():
                continue
            results.append(linedata)
            i += 1
    return results


if __name__ == '__main__':
    names = ['learnplot3_sent_skip',
             'learnplot2_word',
             'learnplot2_sent_glove',
             'learnplot3_sent'
             ]

    accs_in = [import_file(n) for n in names]
    accs = [n[0]['ccuracy'] for n in accs_in]


    labels = ['Skip-gram\n Embedding',
            'Problemspezifisches\n Embedding',
            'GloVe\n Embedding',
            'alle 3 Embeddings',
            ]
    men_means = accs

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    #fig, ax = plt.subplots()
    rects1 = plt.bar(x, men_means, width, label='Accuracy')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Accuracy')
    #plt.title('Scores by group and gender')
    plt.xticks(x, labels)
    plt.yticks(range(50, 100, 10), [f"{e}%" for e in range(50, 100, 10)])
    #plt.xticklabels()
    plt.legend()

    ax = plt.axes()

    axes = plt.gca()
    #axes.set_xlim([0, 100])
    # axes.set_ylim([0, 1])
    axes.set_ylim([50, 90])


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.2f}%'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)

    plt.tight_layout()

    plt.show()
