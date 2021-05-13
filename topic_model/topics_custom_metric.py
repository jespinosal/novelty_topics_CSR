import pandas as pd
from collections import Counter
from topic_model.preprocessing import pre_processing_pipeline

TOPICS_KEY_WORDS = {'environmental': 'renewable energy impact environmental packaging reduce packaging water'
                                     ' plastic environment sustainability sustainable climatic climate weather'
                                     ' temperature',
                    'social': 'human responsibility people social inclusive community employee rights diversity health',
                    'corporate governance': 'mission fiscal right market policy program corporate governance'
                                            ' politics law business'}


def parse_topics_key_words(common_dictionary):
    """
    Transform TOPICS_KEY_WORDS into dictionary of lemmatized tokens of word_ids
    according the dictionary common_dictionary
    :param common_dictionary:
    :return:
    """
    key_words_processed = {key: pre_processing_pipeline(text=words) for key, words in TOPICS_KEY_WORDS.items()}
    key_words_map = {key:common_dictionary.doc2bow(word_tokens) for key,word_tokens in key_words_processed.items()}

    word_ids_per_topic ={}
    for topic_name in key_words_map.keys():
        word_ids_per_topic[topic_name] = [word_id for word_id,word_freq in key_words_map[topic_name]]
    return word_ids_per_topic


def lda_models_evaluator(model_list, k_values, coherence, common_dictionary):
    """
    Evaluate the lda_models available in model_list.
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
    topics_probs_df['k_values'] =k_values
    topics_probs_df['avg_prob'] = topics_probs_df.mean(axis=1)
    topics_probs_df['coherence'] = coherence
    topics_probs_df['custom_metric'] = topics_probs_df['avg_prob']*topics_probs_df['coherence']

    return topics_probs_df


def map_lda_topics(best_model, common_dictionary):
    """
    Categorizing topics into TOPICS_KEY_WORDS dimensions:
    'environmental, social and corporate
    :param best_model:
    :return:
    """
    word_ids_per_topic = parse_topics_key_words(common_dictionary)

    topic_probs = {}
    for topic_name, word_ids in word_ids_per_topic.items():
        word_probs = []
        for word_id in word_ids:
            word_probs.extend(best_model.get_term_topics(word_id, minimum_probability=0.001))
        topic_lda_ids = [topic_lda_id for topic_lda_id, probs in word_probs]
        topic_probs[topic_name] = Counter(topic_lda_ids)

    return pd.DataFrame(topic_probs)


"""
import pandas as pd
from topic_model.preprocessing import pre_processing_pipeline

TOPICS_KEY_WORDS = {'environmental': 'environment climatic climate weather temperature',
                    'social': 'community employee human rights diversity',
                    'corporate governance': 'governance politics law'}

def parse_topics_key_words()
    
    
def lda_models_evaluator(common_dictionary,model_list, k_values, coherence):


    :param common_dictionary:
    :param model_list:
    :param k_values:
    :param coherence:
    :return:


    key_words_processed = {key: pre_processing_pipeline(text=words) for key, words in TOPICS_KEY_WORDS.items()}
    key_words_map = {key:common_dictionary.doc2bow(word_tokens) for key,word_tokens in key_words_processed.items()}

    word_ids_per_topic ={}
    for topic_name in key_words_map.keys():
        word_ids_per_topic[topic_name] = [word_id for word_id,word_freq in key_words_map[topic_name]]

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
    topics_probs_df['k_values'] =k_values
    topics_probs_df['avg_prob'] = topics_probs_df.mean(axis=1)
    topics_probs_df['coherence'] = coherence
    topics_probs_df['custom_metric'] = topics_probs_df['avg_prob']*topics_probs_df['coherence']

    return topics_probs_df

def get_main_topics(best_model):
    best_model

"""
