from basic.main import *

model = ExpEmb(corpus_file="../inputs/citeseer_node2vec2.corpus",
               embed_file="../outputs/exp_emb_citeseer_node2vec.embedding",
               num_processes=1, method_name='bernoulli',
               dim=128, neg_samples=5, init_alpha=0.025, win=10, min_count=0)
model.train()
