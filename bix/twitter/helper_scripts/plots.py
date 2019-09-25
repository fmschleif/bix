import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import label
from pylab import rcParams
rcParams['figure.figsize'] = 7, 3.8


def import_file(path):
    results = []
    i = 0
    with open(path, 'r') as infile:
        for line in infile.readlines():
            linedata = {}
            for match in re.findall("([a-z_]*?): (\d+\.\d+)", line):
                linedata[match[0]] = float(match[1])
            #linedata['index'] = i
            if 'loss' not in linedata.keys():
                continue
            results.append(linedata)
            i += 1
    return results


if __name__ == '__main__':
    names = ['learnplot3_sent', 'learnplot3_sent_skip', 'learnplot2_word', 'learnplot2_sent_glove']



    for n in names:
        normal = import_file(n)
        #print(normal)
        loss = [d['loss'] for d in normal]
        acc = [d['acc'] for d in normal]
        val_loss = [d['val_loss'] for d in normal]
        val_acc = [d['val_acc'] for d in normal]
        indices = list(range(50))

        # plotting the points
        #plt.plot(indices, loss, label='loss')
        #plt.plot(indices, val_loss, label='validation loss')

        plt.plot(indices, acc, label='accuracy')
        plt.plot(indices, val_acc, label='validation accuracy')

        # naming the x axis
        plt.xlabel('epoch')
        # naming the y axis
        #plt.ylabel('Genauigkeit')
        plt.tight_layout()

        plt.legend()

        axes = plt.gca()
        axes.set_xlim([0, 50])
        #axes.set_ylim([0, 1])
        axes.set_ylim([0.5, 1])

        # giving a title to my graph
        #plt.title('My first graph!')


        # function to show the plot
        #plt.show()
        plt.savefig(n + '_acc')
        # print(normal)

        plt.clf()
        # plotting the points
        plt.plot(indices, loss, label='loss')
        plt.plot(indices, val_loss, label='validation loss')

        #plt.plot(indices, acc, label='accuracy')
        #plt.plot(indices, val_acc, label='validation accuracy')

        # naming the x axis
        plt.xlabel('epoch')
        # naming the y axis
        # plt.ylabel('Worthäufigkeit')
        plt.tight_layout()

        plt.legend()

        axes = plt.gca()
        axes.set_xlim([0, 50])
        axes.set_ylim([0, 1])
        #axes.set_ylim([0.5, 1])

        # giving a title to my graph
        # plt.title('My first graph!')

        # function to show the plot
        #plt.show()
        plt.savefig(n + '_loss')
        plt.clf()
