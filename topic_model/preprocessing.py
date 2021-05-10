import string
import spacy as sp
import nltk
from nltk.corpus import stopwords

nlp = sp.load('en_core_web_sm', disable=['ner', 'parse'])

# nltk.download('stopwords')
# python -m spacy download en_core_web_sm

CUSTOM_STOP_WORDS = ['http'] + [letter for letter in string.ascii_lowercase]


def delete_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def convert_to_lower(sentence):
    """
    Lower case for the text documents
    :param sentence:
    :return:
    """
    return sentence.lower()


def word_tokenizer(text):
    """
    Split the string text in tokens
    :param text:
    :return:
    """

    tokens = nltk.tokenize.word_tokenize(text, language='english')

    return tokens


def custom_stop_words_deletion(text_tokens):
    "Delete stop words from text_tokens"
    base = stopwords.words('english')
    stopwords_list = CUSTOM_STOP_WORDS + base
    return [text for text in text_tokens if text not in stopwords_list]


# Source: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#9createbigramandtrigrammodels
def get_lemma(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Method to get lemmas of words with pos tag indicated on allowed_postags"""
    # WARN: Lemma needs complete phrases to work better and punctuation.
    texts_tokens_lemma = []
    doc = nlp(text)
    texts_tokens_lemma.extend([token.lemma_ if token.pos_ in allowed_postags
                      else token.text for token in doc])
    return texts_tokens_lemma

def pre_processing_pipeline(text):
  text = convert_to_lower(sentence= text)
  text = delete_punctuation(text)
  text_tokens = get_lemma(text)
  text_tokens = custom_stop_words_deletion(text_tokens)
  return text_tokens

