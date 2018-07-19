from gensim.models.word2vec import *



class Word2VecWrapper(Word2Vec):
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False):

        super(Word2VecWrapper, self).__init__(sentences, size, alpha, window, min_count,
                                              max_vocab_size, sample, seed, workers, min_alpha,
                                              sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                              trim_rule, sorted_vocab, batch_words, compute_loss)


