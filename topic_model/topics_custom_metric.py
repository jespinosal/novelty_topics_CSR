import pandas as pd
from collections import Counter
from topic_model.preprocessing import pre_processing_pipeline
from config.config import TOPICS_KEY_WORDS

"""
    The script aims to find the best model in list of candidate models available on "model_list".

    Each candidate model was generate with a different number of topics (building 100 models, from 2 to 101 topics).

    TheTOPICS_KEY_WORDS is a dictionary that define the main words to consider in each topic according a human expert. I
    defined a first set of candidate words according what I read in a sample of reports. The words in this dictionary
    will help us to identify which "model topics" are important and which not. (Remember each LDA topic is composed
    by a set of weighted words. For instances -> topic_0 = ('world':0.041), ('peace':0.033) ..('love':0.013)


    We have two types of topics:
    - hypothetical topics: the wanted topics that we want to find, according the business/research needs.
    - model topics: the topics the lda model was able to find (23 topics for the best model according last experiment)

    Once we choose the best model on "model_list", we will analize each topic in the model to decide which models
    are useful. We will try to match "hypothetical topics" with "model topics". For instance:
    "environmental" : topic_3, topic_5 ... topic_23
    "social": topic_1, topic_13 ... topic_23
    "corporate governance": topic_1, topic_13 ... topic_23
    "unuseful": topic_1, topic_13 ... topic_23



    We used to metrics to choose the best model on model_list.
    1) Metric "avg_prob" : use the TOPICS_KEY_WORDS to find the environmental

    The group of words defined on TOPICS_KEY_WORDS are extracted in

    This method evaluates the candidate models available in model_list, where each candidate was generated using
    a different k_value (number of topics per model).
    This method analyze the word probabilities of each topic in each model candidate, computing the words of
    TOPICS_KEY_WORDS with minimum_probability>=0.001.
    The main idea is choose the model that maximize the probabilities values in the wanted words
    defined on TOPICS_KEY_WORDS, to choose de model in model_list that contatins topics with higher probabilities.

    - The column "avg_prob" computes de word probabilities TOPICS_KEY_WORDS for
    - The metric

    The lda_topics_performance contains the follow columns where the meaining of each row is:
    enviromental: average probability
    social:
    corporate governance:
    k_values
    avg

"""


def parse_topics_key_words(common_dictionary):
    """
    Transform TOPICS_KEY_WORDS into dictionary of lemmatized tokens of word_ids
    according the dictionary common_dictionary
    :param common_dictionary:
    :return:
    """
    key_words_processed = {key: list(set(pre_processing_pipeline(text=words))) for key, words in
                           TOPICS_KEY_WORDS.items()}
    key_words_map = {key: common_dictionary.doc2bow(word_tokens) for key, word_tokens in key_words_processed.items()}

    word_ids_per_topic = {}
    for topic_name in key_words_map.keys():
        word_ids_per_topic[topic_name] = [word_id for word_id, word_freq in key_words_map[topic_name]]
    return word_ids_per_topic


def lda_models_evaluator(model_list, k_values, coherence, common_dictionary):
    """


    :param common_dictionary:
    :param model_list:
    :param k_values:
    :param coherence:
    :return:
    """

    word_ids_per_topic = parse_topics_key_words(common_dictionary)

    word_probs_topics = {}
    for model_id, lda_model in enumerate(model_list):
        topic_probs = {}
        for topic_name, word_ids in word_ids_per_topic.items():
            word_probs = []
            for word_id in word_ids:
                word_probs.extend(lda_model.get_term_topics(word_id, minimum_probability=0.001))
            topic_probs[topic_name] = sum([probs for topic_lda_id, probs in word_probs])
        word_probs_topics[model_id] = topic_probs

    topics_probs_df = pd.DataFrame(word_probs_topics).transpose()
    topics_probs_df['k_values'] = k_values
    topics_probs_df['avg_prob'] = topics_probs_df.mean(axis=1)/topics_probs_df['k_values']
    topics_probs_df['coherence'] = coherence
    topics_probs_df['custom_metric'] = topics_probs_df['avg_prob']*topics_probs_df['coherence']

    return topics_probs_df


def map_lda_topics(best_model, common_dictionary):
    """
    Categorizing topics into TOPICS_KEY_WORDS dimensions : 'environmental, social and corporate
    Each k topic will be represented as weight distribution of ESC categories.
    The method will count the amount of topics in every word of ESC dictionary.
    :param best_model:
    :return:
    """
    word_ids_per_topic = parse_topics_key_words(common_dictionary)

    topic_probs = {}
    for topic_name, word_ids in word_ids_per_topic.items():
        word_probs = []
        for word_id in word_ids:
            # get word_ids probs in every k-topic with minimum_probability
            word_probs.extend(best_model.get_term_topics(word_id, minimum_probability=0.001))
            # get the k-topics that match with word_id of every ESC category
            # best_model.get_term_topics(1, minimum_probability=0.001)
            # --> [(49, 0.0039986055), (68, 0.001230073), (71, 0.0013883046), (89, 0.0012172041)]
        topic_lda_ids = [topic_lda_id for topic_lda_id, probs in word_probs]
        # word_probs accumulate the topics occurrences of every word in each ESC category
        topic_probs[topic_name] = Counter(topic_lda_ids)
        # topic_probs get the frequency of occurrences of k topics for each word of ESC dictionary
        # --> {'environmental':{46: 24, 82: 24 ... 'corporate governance': {25: 21...

    return pd.DataFrame(topic_probs)

