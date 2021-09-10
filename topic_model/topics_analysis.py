"""
This module will load the best LDA model and will implement the following steps:
1. Label each topic according the mosd dominant categories
2. Filer the non usefull topics
3. Implement a innovation index for topics
4. Summarize the index at company level
4. Sort and present the results in a table

"""
import pandas as pd
import operator
import numpy as np
from topic_model.lda_model import map_lda_topics, get_corpus
from gensim.models.ldamulticore import LdaMulticore
from config.config import LDA_PATHS


def fill_in_empty_equivalences(lda_and_issue_weights, k=105):
    """
    Some LDA topics do not match with any the words on TOPICS_KEY_WORDS.
    The table lda_and_issue_weights do not include those equivalences.
    This method will include the empty equivalences on lda_and_issue_weights
    imputing weights with the value 0. (adding new rows with 0)
    The
    :return:
    """
    missing_lda_topics_ids = set(range(0, k)).difference(set(ESG_weights.index))

    for missing_lda_topics_id in missing_lda_topics_ids:
        row = pd.Series({'environmental': 0, 'social': 0, 'corporate governance':0}, name=missing_lda_topics_id)
        lda_and_issue_weights = lda_and_issue_weights.append(row)

    return lda_and_issue_weights


def build_lda_topic_vector(company_topics: list, k: int = 105) -> np.array:
    """
    This method calculate the mean weight of topics in company_topics
    :param company_topics: list of tuples [(topic_id, topic_weight_in_company_report)
    -> example: [(22, 0.043131966), (26, 0.029636152), (34, 0.063039675), (45, 0.037291616)
    :param k: number of topics in LDA model
    :return: A numpy array with the mean of topics weights per company. Explains the topic
    contribution in each company.
    -> example:
    """
    topic_wights = np.zeros(k)
    topic_freq = np.zeros(k)
    for topic, weight in company_topics:
        topic_wights[topic] += weight
        topic_freq[topic] += 1
    topic_avg_weights = topic_wights/topic_freq
    # some topics with 0 freq will produce NaN values that must be imputed to 0
    return np.nan_to_num(topic_avg_weights)


def lda_topics_to_ESG(lda_and_issue_weights):
    """
    lda_and_issue_weights is a table where the columns are the CSR issue categories enviroment, social and
    corporate_governance. The table has k-topics (105) rows where each row contains the contribution
    of every CSR issue categories in each k topic,
    -> example:
    k-topic environmental social  corporate governance
    0             8.0     9.0                   9.0
    42           11.0    10.0                  12.0
    65            6.0     4.0                   3.0
    This methods calculates the topic equivalences of each LDA topic in terms of the CSR categories.
    The lda_and_issue_weights is processed to get CSR issue field with higher weight per row.
    :param lda_and_issue_weights:
    :return: a dictionary with the format {'lda_topic_id': 'csr_topic_name'}
    --> example: {0: 'social', 42: 'corporate governance', 65: 'environmental' ..
    """
    return lda_and_issue_weights.idxmax(axis=1).to_dict()


def esg_company_contribution(company_topics_vector, lda_and_issue_weights):
    """
    This method computes the topics weights and esg weigths to get the overall esg contribution
    in the each company.
    company_topics_vector
    :param company_topics_vector:
    :param lda_and_issue_weights:
    :return:
    """
    esg_computed_weights = {'environmental': 0.0,
                            'social': 0.0,
                            'corporate governance': 0.0}
    lda_and_issue_weights = lda_and_issue_weights

    for topic_id, topic_weight in enumerate(company_topics_vector):
        computed_weights = lda_and_issue_weights.loc[topic_id]*topic_weight
        computed_weights = computed_weights.to_dict()
        esg_computed_weights['environmental'] += computed_weights['environmental']
        esg_computed_weights['social'] += computed_weights['social']
        esg_computed_weights['corporate governance'] += computed_weights['corporate governance']
        # if you want to normalize to 0 to 1 the final coefficients
        #  total = sum(esg_computed_weights.values())
        #  esg_computed_weights = {esg:weight/total for esg, weight in esg_computed_weights.items()}
    return esg_computed_weights


if __name__ == "__main__":

    # read best found model k=105
    lda_model = LdaMulticore.load(LDA_PATHS['lda_model'])
    # calculate ESG topics contribution on K-LDA topics (df index means the k_topic_id)
    ESG_weights = map_lda_topics(best_model=lda_model, common_dictionary=lda_model.id2word)
    # Weights with NA means the topic has not any word with contribution over minimum_probability=0.001
    ESG_weights = ESG_weights.fillna(0)
    # Impute missing equivalences
    ESG_weights = fill_in_empty_equivalences(lda_and_issue_weights=ESG_weights, k=105)
    # Normalize (min-max) to get values from 0 to 1
    ESG_weights = ESG_weights.div(ESG_weights.sum(axis=1), axis=0)
    ESG_weights = ESG_weights.fillna(0)
    print(ESG_weights)  # --> ESC contribution in each k-topic, normalized
    # Overall efforts in each ESG field (mean of freq ESC in K-topics)
    overall_ESG_efforts = ESG_weights.sum(axis=0)/ESG_weights.sum(axis=0).sum()
    print(overall_ESG_efforts)


    # Show most suitable topics equivalences considering max value:
    topic_to_ESG_dict = lda_topics_to_ESG(lda_and_issue_weights=ESG_weights)
    print(topic_to_ESG_dict)

    # Read company reports dataset and assign the corresponding topics in each report
    data_frame_corpus = pd.read_csv(LDA_PATHS['text_corpus'], sep=';')
    data_frame_corpus = data_frame_corpus[~data_frame_corpus['corpus'].isna()]  # Filter None NaN docs
    corpus, _ = get_corpus(data_frame_corpus, common_dictionary=lda_model.id2word)
    # -> Get the topics over the threshold minimum_probability
    topics_per_report = [lda_model.get_document_topics(report_corpus,
                                                       minimum_probability=0.02) for report_corpus in corpus]
    data_frame_corpus['topics'] = topics_per_report
    # -> Summarize the results at company level:
    df_company_topics = data_frame_corpus.groupby(['company_name']).agg({'topics': 'sum'}).reset_index()
    # -> Build a vector k-topics contribution per company
    df_company_topics['topics_vector'] = df_company_topics.topics.apply(lambda row:
                                                                        build_lda_topic_vector(company_topics=row,
                                                                                               k=lda_model.num_topics))
    # -> Get max values at lda topic level
    df_company_topics['max_lda_topic'] = df_company_topics.topics_vector.apply(lambda row: row.argmax())
    df_company_topics['max_lda_weight'] = df_company_topics.apply(lambda row:
                                                                  row['topics_vector'][row['max_lda_topic']], axis=1)

    # Compute ESG contribution on LDA topic vectors
    df_company_topics['esg_vector'] = df_company_topics.topics_vector.apply(lambda row:
                                                                            esg_company_contribution(row,
                                                                                                     ESG_weights))
    # -> Get max values at esg topic level
    df_company_topics['max_esg_topic'] = df_company_topics.esg_vector.apply(lambda row: max(row.items(),
                                                                                            key=operator.itemgetter(1))[0])
    df_company_topics['max_esg_weight'] = df_company_topics.esg_vector.apply(lambda row: max(row.items(),
                                                                                            key=operator.itemgetter(1))[1])

    # -> Save df_company_topics dataframe:
    df_company_topics.to_csv(LDA_PATHS['company_topics_efforts'], sep=';')
    # -> Get companies with more importance per esg topic
    df_environmental = df_company_topics[df_company_topics.max_esg_topic.isin(['environmental'])].sort_values(by=['max_esg_weight'],
                                                                                                ascending=False)
    df_social = df_company_topics[df_company_topics.max_esg_topic.isin(['social'])].sort_values(by=['max_esg_weight'],
                                                                                                ascending=False)
    df_corporate_governance = df_company_topics[df_company_topics.max_esg_topic.isin(['corporate governance'])].sort_values(by=['max_esg_weight'],
                                                                                                ascending=False)


    print(f"top 10 companies most effort on environmental {[df_environmental[['company_name','max_esg_weight']][0:10]]}")
    print(f"top 10 companies most effort on social {[df_social[['company_name', 'max_esg_weight']][0:10]]}")
    print(f"top 10 companies most effort on corporate governance {[df_corporate_governance[['company_name', 'max_esg_weight']][0:10]]}")

    """
    top 10 companies most effort on environmental 
                 company_name  max_esg_weight
    163          JNJ        2.196900
    185          MMM        1.988008
    38           AZA        1.912170
    137          HAL        1.881263
    254         UMGZ        1.807874
    157          IPG        1.751272
    152          IFF        1.667992
    204           OC        1.477730
    260           WM        1.436173
    6            AAL        1.331152]
    top 10 companies most effort on social 
    253          TWX        1.962934
    121          FLR        1.051651
    84          CSCO        1.027087
    146          HLT        0.816335
    50            BF        0.790706
    255          UNM        0.763049
    139         HBAN        0.736173
    11          ADBE        0.668154
    59           BSX        0.622122
    36           AVY        0.611947]
    top 10 companies most effort on corporate governance 
    10           ABX        2.021244
    133          GPS        1.430660
    170          KMB        1.417466
    233          ROH        1.415601
    24           ALL        1.378015
    158          IQV        1.293953
    270          XRX        1.228888
    159           IR        1.226705
    120         FITB        1.200475
    96           DIS        1.172478]
    """


