import io
import numpy as np
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage


def pdf_miner_reader(pdf_path, page_indexes):
    """
    This fuction read a PDF document page by page using pdfmine library
    :param pdf_path:  directory that contains a pdf file
    :param page_indexes:
    :return: It return a iterator, to read page by page the data
    """
    text_raw_corpus = []
    with open(pdf_path, 'rb') as fh:
        pages = list(PDFPage.get_pages(fh, caching=True, check_extractable=True))
        page_positions = np.array(range(0, len(pages))).take(page_indexes)
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


if __name__ == "__main__":

    file_path_ = 'report_files/microsoft/Microsoft-2019-CSR-Annual-Report.pdf'
    page_indexes_ = [1, 2, 3, -3, -2, -1]
    raw_corpus_text_miner = pdf_miner_reader(file_path_, page_indexes_)






