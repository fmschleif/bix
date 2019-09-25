#word:
#love - mean: 0.8737647533416748, avg: 0.8737647533416748, acc: 0.884101927280426
#sad - mean: 0.17584118247032166, avg: 0.17584118247032166, acc: 0.16147859394550323

#all:
#love - mean: 0.8909766674041748, avg: 0.8909766674041748, acc: 0.8964112401008606
#sad - mean: 0.14818458259105682, avg: 0.14818458259105682, acc: 0.13813228905200958




import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Median', 'Arithmetisches Mittel', 'Genauigkeit']

word_love = [0.8737647533416748, 0.8737647533416748, 0.884101927280426]
word_sad = [0.17584118247032166, 0.17584118247032166, 1-0.16147859394550323]
all_love = [0.8909766674041748, 0.8909766674041748, 0.8964112401008606]
all_sad = [0.14818458259105682, 0.14818458259105682, 1-0.13813228905200958]


#men_means = [20, 34, 30, 35, 27]
#women_means = [25, 32, 34, 20, 25]

for x1, x2, name in [(word_love, all_love, 'Analyse des Hashtags "#love"'),
                     (word_sad, all_sad, 'Analyse des Hashtags "#sad"')]:

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, x1, width, label='Problemspezifisches Embedding')
    rects2 = ax.bar(x + width/2, x2, width, label='Alle 3 Embeddings')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Scores')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    axes = plt.gca()
    # axes.set_xlim([0, 100])
    # axes.set_ylim([0, 1])
    axes.set_ylim([0, 1.2])


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
