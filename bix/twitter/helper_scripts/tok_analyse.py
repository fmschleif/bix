from bix.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer

import matplotlib.pyplot as plt

if __name__ == '__main__':
    t = load_tokenizer('learn')
    print(f'indexsize: {len(t.word_counts)}')
    # print('wordcounts')
    # print(t.word_counts)
    # print('document_count')
    # print(t.document_count)
    # print('wordindex')
    # print(t.word_index)
    # print('word_docs')
    # print(t.word_docs)

    for e, n in sorted(t.word_counts.items(), key=lambda x: x[1])[-20:]:
        print('\t' + str(e) + ': ' + str(n))

    mini = [v for k,v in t.word_counts.items() if v > 5]
    print(f'n: {len(mini)}')

    mini = list(reversed(sorted(mini)))[:30000]

    print(f'30000: {mini[-1]}')
    print(f'25000: {mini[25000]}')
    print(f'20000: {mini[20000]}')
    print(f'15000: {mini[15000]}')
    print(f'10000: {mini[10000]}')
    print(f'5000: {mini[5000]}')

    # x axis values
    mini = enumerate(mini)
    x, y = map(list, zip(*mini))

    # corresponding y axis values
    #y = v

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Wortindex')
    # naming the y axis
    plt.ylabel('Worthäufigkeit')

    axes = plt.gca()
    axes.set_xlim([0, 30000])
    axes.set_ylim([0, 20000])

    # giving a title to my graph
    #plt.title('My first graph!')

    # function to show the plot
    plt.show()
    #plt.savefig('wort_test')


