import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array
import networkx as nx
from scipy.sparse import *


class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items  # List of VocabItem objects
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print('Total words in training file: %d' % self.word_count)
        print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab, power = 0.75):
        vocab_size = len(vocab)

        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        # table_size = 1e8 # Length of the unigram table
        table_size = np.uint32(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0.0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

        print("Unigram table construction has just finished!")

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


class Utils:
    def __init__(self):
        pass

    def sigmoid(z):
        if z > 6:
            return 1.0
        elif z < -6:
            return 0.0
        else:
            return 1 / (1 + math.exp(-z))

    def log_sigmoid(z):

        return -math.log(1.0 + math.exp(-z))


def _init_process(self, *args):
    #pass
    #global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    #global win, num_processes, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    #fi = open(self._corpus_file, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
    #    self._syn0 = np.ctypeslib.as_array(syn0_tmp)
    #    self._syn1 = np.ctypeslib.as_array(syn1_tmp)



global num_processes, vocab, vocab_size, global_word_count, alpha, fi, win, method, dim, table
global neg_sample_count, syn0, syn1, dim, min_count

def initialize(corpus_file):

    # Read train file to init vocab
    vocab = Vocab(corpus_file, min_count)
    vocab_size = len(vocab)

    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    # Global word count
    global_word_count = Value('i', 0)
    # Initializing unigram table
    table = UnigramTable(vocab)

    fi = open(corpus_file, 'r')
    print("The system has been activated, ready for launching!")




def train_process(pid):

    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    fi.seek(start)
    # print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < alpha * 0.0001: alpha = alpha * 0.0001

                # Print progress info
                sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?

            if method == "bernoulli":

                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg_sample_count > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg_sample_count)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = Utils.sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]              # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word] # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e
                """
                """
            else:
                for context_word in context:
                    # Init neule with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg_sample_count > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg_sample_count)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        diff = syn0[context_word] - syn1[target]
                        neg_term = 1.0 / (np.exp(np.dot(diff, diff)/2.0) - 1.0)
                        if label == 1:
                            g = diff
                        else:
                            g = -diff * neg_term
                        g = alpha * g
                        neu1e += -g
                        syn1[target] += g

                    # Update syn0
                    syn0[context_word] += neu1e


                """ 
                nbnb = generate_nn_matrix(vocab)
                for context_word in context:
                    # Init neule with zeros
                    neu1e = np.zeros(dim)
    
                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        # print(vocab.indices([context_word]))
                        prod = np.dot(syn0[context_word], syn1[target]) + 1e-6
                        g = 1.0 - (float(nbnb[vocab.indices([context_word])[0], vocab.indices([target])[0]]) / prod)
                        g = alpha * g
                        neu1e += g * syn1[target]
                        syn1[target] += g * syn0[context_word]
    
                    # Update syn0
                    syn0[context_word] += neu1e
                """

            word_count += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()


def train():

    # Begin training using num_processes workers
    t0 = time.time()
    #pool = Pool(processes=self._num_processes, initializer=_init_process,
    #            initargs=(self._vocab, self._syn0, self._syn1, self._table, self._dim, self._alpha,
    #                      self._win, self._num_processes, self._global_word_count, self._embed_file))
    pool = Pool(processes=num_processes)
    pool.map(train_process, range(num_processes))
    t1 = time.time()

    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')

    # Save model to file
    save(vocab, embed_file)


def save(vocab, fo):
    print('Saving model to', fo)
    dim = len(syn0[0])

    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0)-3, dim))
    for token, vector in zip(vocab, syn0):
        word = token.word
        if word not in ['<bol>', '<eol>', '<unk>']:
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()





if __name__ == '__main__':

    argsfi = "./citeseer_n80_l10_w10_k80_deepwalk_node_corpus.corpus"
    argsfo = "./output_exp2.embedding"
    argscbow = 0
    argsneg = 5
    argsdim = 128
    argsalpha = 0.025
    argswin = 10
    argsmin_count = 0
    argsnum_processes = 1
    argsbinary = 0

    corpus_file = "../inputs/citeseer_node2vec2.corpus"
    embed_file = "../outputs/exp_emb_citeseer_node2vec.embedding"
    num_processes = 1
    method_name = 'bernoulli'
    dim = 128
    neg_samples = 5
    alpha = 0.025
    win = 10
    min_count = 0

    initialize(corpus_file=corpus_file)
    train()
