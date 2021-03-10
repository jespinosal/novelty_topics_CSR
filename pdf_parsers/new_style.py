import PyPDF4


def pdf_new_style_reader():
    pass
    return " "


if __name__ == "__main":

    file_path_ = 'report_files/microsoft/Microsoft-2019-CSR-Annual-Report.pdf'
    page_indexes_ = [1, 2, 3, -3, -2, -1]
    raw_corpus = pdf_new_style_reader(file_path_, page_indexes_)
