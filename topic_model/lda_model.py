import random
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from topic_model.preprocessing import pre_processing_pipeline
from gensim.models.ldamulticore import LdaMulticore
from topic_model.topics_custom_metric import lda_models_evaluator, map_lda_topics


MODEL_PATH = 'results'
SEED = 100
random.seed(SEED)


def get_corpus(data_frame_corpus, common_dictionary=None):
    """
    Build the text corpus using the common_dictionary, if it is None, it will be generated
    :param data_frame_corpus:
    :param common_dictionary:
    :return:
    """

    # clean and report empty reports
    print('The following files cannot be read')
    print(data_frame_corpus[data_frame_corpus['corpus'].isna()]['company_files'])
    data_frame_corpus = data_frame_corpus.copy()
    data_frame_corpus = data_frame_corpus[~data_frame_corpus['corpus'].isna()]

    # docs corpus preprocessing
    print('Pre-processing corpus')
    data_frame_corpus['corpus_tokens'] = data_frame_corpus.corpus.apply(lambda text: pre_processing_pipeline(text=text))

    if common_dictionary is None:
        # encapsulates the mapping between normalized words and their integer ids -> this map have to be saved in disc
        common_dictionary = Dictionary(data_frame_corpus.corpus_tokens.to_list())

    # Transform corpus in bag of word format (token_id, token_count)
    print('Building corpus')
    corpus = [common_dictionary.doc2bow(token_list) for token_list in data_frame_corpus['corpus_tokens'].to_list()]

    return corpus, common_dictionary, data_frame_corpus


def lda_topic_model_tuning(text_corpus_file_path):

    start_time = datetime.now()
    # Load de report corpus dataframe
    data_frame_corpus = pd.read_csv(text_corpus_file_path, sep=';')

    corpus, common_dictionary, data_frame_corpus = get_corpus(data_frame_corpus, common_dictionary=None)

    # LDA parallelization using multiprocessing CPU cores to parallelize
    k_values = list(range(104, 107))
    coherence = []
    model_list = []
    for topic_n in tqdm(k_values):
        lda_model = LdaMulticore(corpus=corpus,  # data_frame_corpus['corpus_tokens'].to_list()
                                 id2word=common_dictionary,
                                 num_topics=topic_n,
                                 workers=7,
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

    # Save the tuning results, models and parameters
    best_model_id = np.argmax(coherence)
    best_model = model_list[best_model_id]
    best_coherence = coherence[best_model_id]
    best_k = k_values[best_model_id]

    # plot LDA results for coherence score
    print('best_coherence:', best_coherence, 'best_k:', best_k)
    plt.plot(k_values, coherence)
    plt.plot(k_values, coherence, 'bo')
    plt.plot(best_k, best_coherence, 'ro')
    plt.xlabel("Num K Topics")
    plt.ylabel("Coherence score")
    plt.show()
    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')

    return model_list, k_values, coherence, common_dictionary, best_model, corpus


def save_topic_words(best_model, top=30):
    top_words_per_topic = []
    for t in range(best_model.num_topics):
        top_words_per_topic.extend([(t,) + x for x in best_model.show_topic(t, topn=top)])

    top_words = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'Prob'])
    top_words.to_csv(os.path.join(MODEL_PATH, 'top_words.csv'), sep=';')
    return top_words


# Analysis of LDA models
class LDAModel:

    def __init__(self, text_corpus_file_path):
        self.text_corpus_file_path = text_corpus_file_path
        self.model_list = None
        self.k_values = None
        self.coherence = None
        self.common_dictionary = None
        self.best_model = None
        self.corpus = None
        self.lda_topics_performance = None
        self.ESG_topics_distribution = None

    def train(self):
        self.model_list, self.k_values, self.coherence, self.common_dictionary, self.best_model, self.corpus = \
            lda_topic_model_tuning(self.text_corpus_file_path)

    def model_analysis(self):
        self.lda_topics_performance = lda_models_evaluator(self.model_list, self.k_values,
                                                           self.coherence, self.common_dictionary)
        self.ESG_topics_distribution = map_lda_topics(self.best_model,
                                                      self.common_dictionary)

    def save_best_model(self):
        pass

    def load_best_model(self):
        pass


if __name__ == "__main__":

    from config.config import LDA_PATHS

    text_corpus_file_path_ = LDA_PATHS['text_corpus']
    lda_model_mngr = LDAModel(text_corpus_file_path=text_corpus_file_path_)
    lda_model_mngr = LDAModel(text_corpus_file_path=text_corpus_file_path_)
    lda_model_mngr.train()
    lda_model_mngr.model_analysis()
    # save model, world dict and ESG topics weights
    save_topic_words(lda_model_mngr.best_model, top=30)
    lda_model_mngr.best_model.save(LDA_PATHS['lda_model'])
    lda_model_mngr.common_dictionary.save(LDA_PATHS['lda_dict'])
    lda_model_mngr.ESG_topics_distribution.to_csv(LDA_PATHS['topics_distribution'], sep=';')
    lda_model_mngr.lda_topics_performance.to_csv(LDA_PATHS['lda_model_performance'], sep=';')







