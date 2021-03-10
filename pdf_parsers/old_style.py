import numpy as np
import PyPDF2


def pdf_old_style_reader(file_path, page_indexes):
    """
    Read a PDF file and build a raw text corpus according the page_indexes
    :param file_path:
    :param page_indexes:
    :return:
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    page_len = pdf_reader.numPages
    page_positions = np.array(range(0, page_len)).take(page_indexes)
    text_raw_corpus = [pdf_reader.getPage(page_position).extractText() for page_position in page_positions]
    return " ".join(text_raw_corpus)


if __name__ == "__main":

    file_path_ = 'report_files/microsoft/Microsoft-2019-CSR-Annual-Report.pdf'
    page_indexes_ = [1, 2, 3, -3, -2, -1]
    raw_corpus = pdf_old_style_reader(file_path_, page_indexes_)





