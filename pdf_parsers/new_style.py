import numpy as np
import PyPDF2
import PyPDF4


def pypdf_scraper_(pdf_reader, page_indexes):
    page_len = pdf_reader.numPages
    page_positions = np.array(range(0, page_len)).take(page_indexes)
    text_raw_corpus = [pdf_reader.getPage(page_position).extractText() for page_position in page_positions]
    return " ".join(text_raw_corpus)


def pypdf2_reader(file_path, page_indexes):
    """
    Read a PDF file and build a raw text corpus according the page_indexes
    :param file_path:
    :param page_indexes:
    :return:
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    return pypdf_scraper_(pdf_reader, page_indexes)


def pypdf4_reader(file_path, page_indexes):
    """
    Read a PDF file and build a raw text corpus according the page_indexes
    :param file_path:
    :param page_indexes:
    :return:
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF4.PdfFileReader(pdf_file_obj)
    return pypdf_scraper_(pdf_reader, page_indexes)


if __name__ == "__main__":

    file_path_ = 'report_files/microsoft/Microsoft-2019-CSR-Annual-Report.pdf'
    page_indexes_ = [1, 2, 3, -3, -2, -1]
    raw_corpus_pypdf2 = pypdf2_reader(file_path_, page_indexes_)
    raw_corpus_pypdf4 = pypdf4_reader(file_path_, page_indexes_)
