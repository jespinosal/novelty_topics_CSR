import numpy as np
import PyPDF2
import PyPDF4
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage


def page_index_to_page_position(pdf_size, page_indexes):
    """

    :param pdf_size:
    :param page_indexes:
    :return:
    """
    if page_indexes:
        page_indexes = [page_index for page_index in page_indexes if abs(page_index) < pdf_size]  # Get valid index
    else:
        page_indexes = range(0, pdf_size)
    page_positions = np.array(range(0, pdf_size)).take(page_indexes)  # Transform index in page position
    return page_positions


def pdf_miner_reader(file_path, page_indexes):
    """
    This fuction read a PDF document page by page using pdfmine library
    :param file_path:  directory that contains a pdf file
    :param page_indexes:
    :return: It return a iterator, to read page by page the data
    """
    text_raw_corpus = []
    with open(file_path, 'rb') as fh:
        pages = list(PDFPage.get_pages(fh, caching=True, check_extractable=True))
        pdf_size = len(pages)

        page_positions = page_index_to_page_position(pdf_size, page_indexes)

        for n, page in enumerate(pages):
            if n in page_positions:
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(resource_manager, fake_file_handle)
                page_interpreter = PDFPageInterpreter(resource_manager, converter)
                page_interpreter.process_page(page)
                text = fake_file_handle.getvalue()
                text_raw_corpus.append(text)
    return " ".join(text_raw_corpus)


def pypdf_scraper_(pdf_reader, page_indexes):
    pdf_size = pdf_reader.numPages
    page_positions = page_index_to_page_position(pdf_size, page_indexes)
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
    file_path_ = 'report_files/microsoft/citizenship2004.pdf'
    page_indexes_ = [1, 2, 3, -3, -2, -1]
    raw_corpus_pypdf2 = pypdf2_reader(file_path_, page_indexes_)
    raw_corpus_pypdf4 = pypdf4_reader(file_path_, page_indexes_)
    raw_corpus_text_miner = pdf_miner_reader(file_path_, page_indexes_)

