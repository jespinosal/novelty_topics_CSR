import re
import unicodedata


def text_cleaner(text):
    """
    This function apply a text normalization getting only valid characters. The valid characters are defined in the
    regex function.
    :param text: Text in string format to process
    :return: It return the same text clean
    """
    #  decode to ascii
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    #  remove nonprintable characters using a character class
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    #  remove the "escape" characters
    regex = re.compile(r'[\n\r\t]')
    text = regex.sub(" ", text)
    # filter only text (the scraper join page numbers with words sometimes)
    text = re.sub('[^a-zA-Z ]+', '', text)
    # remove non simple spaces
    text = re.sub('\s\s+', ' ', text)
    # lower case
    text = text.lower()

    return text


if __name__ == "__main__":

    text_cleaner('Environmental sustainability \t resources 32 Reports hub\x0c \n yeah \ncome 999  '
                  'the ñame & % water  hóla hölä  [ ] () 1990? numb3ers in the 9wor9ld9   er8ror ')


