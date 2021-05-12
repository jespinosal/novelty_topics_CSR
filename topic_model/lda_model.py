import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pandas as pd
from topic_model.preprocessing import pre_processing_pipeline
import logging
from gensim.corpora.dictionary import Dictionary
from topic_model.preprocessing import pre_processing_pipeline
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import random
import matplotlib.pyplot as plt
import numpy as np

SEED = 100
random.seed(SEED)

text_corpus_file_path = 'text_corpus/csr_corpus_pages_index0_1_2_-3_-2_-1.csv'

data_frame_corpus = pd.read_csv(text_corpus_file_path, sep=';')

key_words = {'environmental': ['enviroment', 'climatic'],
             'social': ['community', 'employee', 'human rights', 'diversity'],
             'corporate governance': ['goverment']}

# docs corpus preprocessing
data_frame_corpus['corpus_tokens'] = data_frame_corpus.corpus.apply(lambda text: pre_processing_pipeline(text=text))

# encapsulates the mapping between normalized words and their integer ids -> this map have to be saved in disc
common_dictionary = Dictionary(data_frame_corpus.corpus_tokens.to_list())

# Transform corpus in bag of word format (token_id, token_count)
corpus = [common_dictionary.doc2bow(token_list) for token_list in data_frame_corpus['corpus_tokens'].to_list()]

# LDA parallelization using multiprocessing CPU cores to parallelize
k_values = list(range(3, 10))
coherence = []
model_list = []
for topic_n in k_values:
    lda_model = LdaMulticore(corpus=corpus,  # data_frame_corpus['corpus_tokens'].to_list()
                             id2word=common_dictionary,
                             num_topics=topic_n,
                             workers=3,
                             random_state=SEED,
                             chunksize=100,
                             passes=10,
                             # alpha='auto',  #  auto-tuning alpha not implemented in multicore LDA
                             per_word_topics=True)

    coherence_lda = CoherenceModel(model=lda_model,
                                   texts=data_frame_corpus['corpus_tokens'].to_list(),
                                   coherence='c_v')
    model_list.append(lda_model)
    coherence.append(coherence_lda.get_coherence())

best_model_id = np.argmax(coherence)
best_model = model_list[best_model_id]
best_coherence = coherence[best_model_id]
best_k = k_values[best_model_id]

print('best_coherence:', best_coherence, 'best_k:', best_k)
plt.plot(k_values, coherence)
plt.plot(k_values, coherence, 'bo')
plt.plot(best_k, best_coherence, 'ro')
plt.xlabel("Num K Topics")
plt.ylabel("Coherence score")
plt.show()




"""
Pre processig_
1. Lemma
2. token
3. stopw
4. ngram

similarity analysis??
bigram and trigram is not needed
document level? company? year? or both default!

Strategy:
Filter or agg consifering the 
1. get topics with all text and later apply filters
2. get topics per year?
3. One vs all find topics in one company and find in others? not shure

Define naming method according word rules (priority order?) weighs counting? --> use this criteria best topics?
1) environmental 
2) social (community / employee / human rights / diversity 
3) corporate governance
"""