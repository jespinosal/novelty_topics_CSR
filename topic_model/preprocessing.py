import string
import re
import spacy as sp
import nltk
from nltk.corpus import stopwords

nlp = sp.load('en_core_web_sm', disable=['ner', 'parse'])

# nltk.download('stopwords')
# python -m spacy download en_core_web_sm
# ['http', 'www', 'report', 'pdf', 'company', 'product', 'year', 'business', 'reporting']
CUSTOM_STOP_WORDS = ['http', 'www', 'report', 'pdf', 'company', 'product', 'year', 'business', 'reporting'] + \
                    [letter for letter in string.ascii_lowercase]


def delete_punctuation(text):
    """Delete puntuation tokens in text string """
    # it generates white spaces when punctuation is separated from text: "are you ok ?" -> "are you ok  "
    return text.translate(str.maketrans('', '', string.punctuation))


def format_abbreviation(text):
    """
    Source: https://www.programcreek.com/python/?CodeExample=normalize+text
    :param text:
    :return:
    """
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    return text


def format_punctuation(text):
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    return text


def convert_to_lower(sentence):
    """
    Lower case for the text documents
    """
    return sentence.lower()


def remove_redundant_white_spaces(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_tokenizer(text):
    """
    Split the string text in tokens
    """

    tokens = nltk.tokenize.word_tokenize(text, language='english')

    return tokens


def custom_stop_words_deletion(text_tokens):
    """Delete stop words from text_tokens"""
    base = stopwords.words('english')
    stopwords_list = CUSTOM_STOP_WORDS + base
    return [token for token in text_tokens if token not in stopwords_list]


def get_lemma(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Method to get lemmas of words with pos tag indicated on allowed_postags"""
    # WARN: Lemma needs complete phrases to work better and punctuation.
    texts_tokens_lemma = []
    doc = nlp(text)
    texts_tokens_lemma.extend([token.lemma_ if token.pos_ in allowed_postags
                      else token.text for token in doc])
    return texts_tokens_lemma


def pre_processing_pipeline(text):
    """Basic text processing pipeline (1) lowercase, (2) delete punctuation, (3) get lemma (4) stop words"""
    text = convert_to_lower(sentence=text)
    text = delete_punctuation(text)
    text = remove_redundant_white_spaces(text)
    text_tokens = get_lemma(text)
    text_tokens = custom_stop_words_deletion(text_tokens)
    return text_tokens


if __name__ == "__main__":

    sample_texts = ['Here we go, it is my first (1s) time in the moon',
                    '3 years ago I studied # math, and now I study medicine!',
                    'A house is not the same @ that a home',
                    'ArE YoU CrazY? are you ok Jhon ? check the https:www.empty.com/empty',
                    'your name is Karol, K A R O L right? k a r o l',
                    'I was running, jumping and swimming all the day']

    for n, sample_text in enumerate(sample_texts):
        print(f'Original text {n}: {sample_text}')
        print(f'Processed text {n}: {pre_processing_pipeline(sample_text)}')