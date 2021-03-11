
"""
The reported files format have the following rules:
0. The project must include a data directory called <report_files>
1. We need inside <report_files> a folder per company; this folder must include the CSR reports to analyze
2. We have to include the reporting year in the file name using the format report_file_name_<year>.pdf
where year is a YYYY format, for instance: report_CSR_disney_2005.pfd, 2017disneycsrupdate.pdf,
report-2005-CSR-disney.pfd etc.
3. If you have the option, download Overviews reports instead of the entire reports, for instance:
https://us.pg.com/sustainability-reports/ offer both options.

The PDF files will read include files from the year 2000 to 2020, it involves many different
PDF formats (old and new style). Each document will be read through different packages to support the different
pdf formats, the order will respect the following order:  PyPDF4 -> PyPDF2 -> pdfplumber -> pdfminer. The reader
script will apply all the libraries returning as more text content as possible.
"""
import os
import pandas as pd
from pdf_readers import pypdf4_reader, pypdf2_reader, pdf_miner_reader
from text_cleaner import text_cleaner

TEXT_FORMAT = '.txt'
PDF_FORMAT = '.pdf'
DATA_PATH = 'report_files'
START_YEAR = 2000
LAST_YEAR = 2020
YEARS = [str(i) for i in list(range(START_YEAR, LAST_YEAR))]


def get_files_format(path_directory):
    """
    Get from the path_directory all the valid files
    :param path:
    :return:
    """
    valid_formats = [PDF_FORMAT, TEXT_FORMAT]
    files = os.listdir(path_directory)
    files = [file for file in files if os.path.isfile(os.path.join(path_directory, file))]
    valid_files = []
    for valid_format in valid_formats:
        valid_files.extend([file for file in files if file.endswith(valid_format)])
    return valid_files


def get_year_from_file_name(file_name):
    """
    Get the year of the report from the file_name
    :param file_name:
    :return:
    """
    year_ = None
    for year in YEARS:
        if year in file_name:
            year_ = year
            break
    return year_


class ReportsReader:
    """
    Get the text corpus of the file_path using several libraries considering the corpus with more text. This method
    will return the text of the pdf pages indicated on the attribute page_indexes.
    """
    def __init__(self, pdf_path, page_indexes):
        self.corpus = ''
        self.pdf_path = pdf_path
        self.page_indexes = page_indexes

    def pdf_to_text_pdf_miner(self):
        raw_corpus = pypdf2_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def pdf_to_text_pypdf2(self):
        raw_corpus = pypdf4_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def pdf_to_text_pypdf4(self):
        raw_corpus = pdf_miner_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def get_max_corpus(self, corpus_method):
        corpus_size = len(self.corpus.split())
        corpus_candidate = corpus_method()
        corpus_candidate_size = len(corpus_candidate.split())
        print(self.pdf_path, corpus_method.__name__, corpus_candidate_size)
        return corpus_candidate if corpus_candidate_size > corpus_size else self.corpus

    def __call__(self):
        pdf_to_text_methods = [self.pdf_to_text_pdf_miner, self.pdf_to_text_pypdf2, self.pdf_to_text_pypdf4]
        if self.pdf_path.endswith(PDF_FORMAT):
            for pdf_to_text_method in pdf_to_text_methods:
                try:
                    self.corpus = self.get_max_corpus(pdf_to_text_method)
                except Exception as e:
                    print(f'Unexpected error for {self.pdf_path}:', e)
        return self.corpus


if __name__ == "__main__":

    company_names = os.listdir(DATA_PATH)
    company_names = [company_name for company_name in company_names
                     if os.path.isdir(os.path.join(DATA_PATH, company_name))]

    company_files = {company_name: get_files_format(path_directory=os.path.join(DATA_PATH, company_name)) for
                     company_name in company_names}

    company_data = {'company_name': [], 'company_files': []}

    for company_name, company_files in company_files.items():
        company_data['company_name'].extend([company_name]*len(company_files))
        company_data['company_files'].extend(company_files)

    df_reports_data = pd.DataFrame(company_data)
    df_reports_data['year'] = df_reports_data['company_files'].apply(lambda row: get_year_from_file_name(row))
    df_reports_data['file_path'] = df_reports_data.apply(lambda row: os.path.join(DATA_PATH,
                                                                                  row['company_name'],
                                                                                  row['company_files']), axis=1)
    page_indexes = [0, 1, -2, -1]
    df_reports_data['corpus'] = df_reports_data['file_path'].apply(lambda pdf_path: ReportsReader(pdf_path,
                                                                                                  page_indexes)())
